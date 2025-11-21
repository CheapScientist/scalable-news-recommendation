import argparse
import os

import numpy as np
import torch
from scipy.sparse import load_npz
from tqdm import tqdm

DATA_DIR = "data/processed"


def load_data(doc_term_path, vocab_path):
    from scipy.sparse import csr_matrix

    print(f"Loading document-term matrix from {doc_term_path}...")
    X = load_npz(doc_term_path)
    if not isinstance(X, csr_matrix := type(load_npz)):
        X = X.tocsr()
    else:
        X = X.tocsr()
    print("Matrix shape:", X.shape)

    print(f"Loading vocab from {vocab_path}...")
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f]
    print("Vocab size:", len(vocab))

    return X, vocab


def make_doc_ids_csr(X):
    """
    For CSR matrix X, build an array doc_ids of length nnz such that
    doc_ids[n] = row index (document) for the n-th nonzero.
    """
    indptr = X.indptr
    D = X.shape[0]
    nnz = X.nnz
    doc_ids = np.empty(nnz, dtype=np.int64)
    for d in range(D):
        start, end = indptr[d], indptr[d + 1]
        doc_ids[start:end] = d
    return doc_ids


def initialize_params(D, V, K, device, seed=0):
    torch.manual_seed(seed)
    # mixture weights
    pi = torch.full((K,), 1.0 / K, device=device)

    # topic-word distributions theta (K x V), random positive then normalize
    theta = torch.rand((K, V), device=device) + 1e-8
    theta = theta / theta.sum(dim=1, keepdim=True)

    return pi, theta


def em_mixture_multinomial_gpu(
    X,
    K=20,
    max_iter=50,
    tol=1e-4,
    alpha=1e-2,
    seed=0,
    device="cuda",
):
    """
    EM for mixture of multinomials using PyTorch tensors on GPU (or CPU).

    X: csr_matrix (D x V) of counts
    K: number of topics
    alpha: Dirichlet prior on topic-word distributions
    device: "cuda" or "cpu"
    """
    # Decide device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    device = torch.device(device)
    print("Using device:", device)

    X = X.tocsr()
    D, V = X.shape

    # Precompute CSR internals
    indices = X.indices.astype(np.int64)
    data = X.data.astype(np.float32)
    indptr = X.indptr.astype(np.int64)
    nnz = X.nnz
    print("Nonzeros:", nnz)

    # doc_ids for each nonzero
    doc_ids = make_doc_ids_csr(X)

    # Move CSR components to device
    indices_t = torch.from_numpy(indices).to(device)
    data_t = torch.from_numpy(data).to(device)
    indptr_t = torch.from_numpy(indptr).to(device)
    doc_ids_t = torch.from_numpy(doc_ids).to(device)

    pi, theta = initialize_params(D, V, K, device=device, seed=seed)

    log_likelihoods = []

    for it in range(max_iter):
        print(f"\n=== [GPU] EM iteration {it + 1}/{max_iter} ===")

        # E-step: responsibilities gamma[d, k]
        gamma = torch.zeros((D, K), dtype=torch.float32, device=device)
        log_likelihood = 0.0

        log_theta = torch.log(theta + 1e-32)  # (K, V)
        log_pi = torch.log(pi + 1e-32)        # (K,)

        for d in range(D):
            start = indptr[d]
            end = indptr[d + 1]
            if start == end:
                # empty doc
                gamma[d] = 1.0 / K
                continue

            idx = indices_t[start:end]       # (L,)
            cnt = data_t[start:end]          # (L,)

            # log p(x_d | z=k) = sum_w x_dw * log theta_kw
            # log_theta[:, idx] -> (K, L)
            log_px_z = (log_theta[:, idx] * cnt.unsqueeze(0)).sum(dim=1)  # (K,)

            log_joint = log_pi + log_px_z  # (K,)

            m = torch.max(log_joint)
            log_joint_shifted = log_joint - m
            exp_shifted = torch.exp(log_joint_shifted)
            denom = exp_shifted.sum()
            gamma_d = exp_shifted / denom

            gamma[d] = gamma_d
            doc_ll = m + torch.log(denom + 1e-32)
            log_likelihood += doc_ll.item()

        log_likelihoods.append(log_likelihood)
        print(f"Log-likelihood: {log_likelihood:.4f}")

        if it > 0:
            ll_old = log_likelihoods[-2]
            rel_change = (log_likelihood - ll_old) / (abs(ll_old) + 1e-12)
            print(f"Relative change: {rel_change:.6e}")
            if abs(rel_change) < tol:
                print("Converged based on tolerance.")
                break

        # M-step
        Nk = gamma.sum(dim=0)  # (K,)
        pi = Nk / float(D)

        print("Updating topic-word distributions on", device, "...")
        topic_word_counts = torch.zeros((K, V), dtype=torch.float32, device=device)

        # For each nonzero, we know:
        # doc_ids_t[n] = d, word indices_t[n] = w, count data_t[n] = c
        # We want: topic_word_counts[k, w] += gamma[d, k] * c

        # gamma_doc_for_nnz: (nnz, K)
        gamma_n = gamma[doc_ids_t, :]  # (nnz, K)
        # multiply by counts
        gw = gamma_n * data_t.unsqueeze(1)  # (nnz, K)
        gw_T = gw.t()  # (K, nnz)

        for k in range(K):
            topic_word_counts[k].index_add_(0, indices_t, gw_T[k])

        topic_word_counts += alpha
        theta = topic_word_counts / topic_word_counts.sum(dim=1, keepdim=True)

    ll_tensor = torch.tensor(log_likelihoods, dtype=torch.float64)
    return (
        pi.detach().cpu().numpy(),
        theta.detach().cpu().numpy(),
        ll_tensor.detach().cpu().numpy(),
    )


def print_top_words(theta, vocab, top_n=10):
    K, V = theta.shape
    for k in range(K):
        top_idx = np.argsort(theta[k])[::-1][:top_n]
        words = [vocab[j] for j in top_idx]
        print(f"Topic {k}: {' '.join(words)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_term", type=str, default=f"{DATA_DIR}/doc_term.npz")
    parser.add_argument("--vocab", type=str, default=f"{DATA_DIR}/vocab.txt")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda", help='"cuda" or "cpu"'
    )
    parser.add_argument("--output", type=str, default=f"{DATA_DIR}/em_gpu_k20.npz")

    args = parser.parse_args()

    from scipy.sparse import load_npz as _ln
    X, vocab = load_data(args.doc_term, args.vocab)

    pi, theta, ll = em_mixture_multinomial_gpu(
        X,
        K=args.K,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
    )

    print("\nFinal log-likelihoods:")
    for i, v in enumerate(ll):
        print(f"Iter {i}: {v:.4f}")

    print("\nTop words per topic (GPU):")
    print_top_words(theta, vocab, top_n=10)

    out_path = args.output
    if out_path.endswith("em_gpu_k20.npz"):
        out_path = f"{DATA_DIR}/em_gpu_k{args.K}.npz"

    print(f"\nSaving GPU parameters to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, pi=pi, theta=theta, ll=ll)

    print("Done (GPU EM).")


if __name__ == "__main__":
    main()
