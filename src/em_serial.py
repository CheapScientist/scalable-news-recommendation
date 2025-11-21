import argparse
import numpy as np
from scipy.sparse import load_npz
import os

DATA_DIR = "data/processed"

def load_data(doc_term_path, vocab_path):
    print(f"Loading document-term matrix from {doc_term_path}...")
    X = load_npz(doc_term_path).tocsr()
    print("Matrix shape:", X.shape)

    print(f"Loading vocab from {vocab_path}...")
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f]
    print("Vocab size:", len(vocab))

    return X, vocab

def initialize_params(D, V, K, seed=0):
    rng = np.random.default_rng(seed)
    # mixture weights
    pi = np.ones(K) / K

    # topic-word distributions (K x V), random positive then normalize
    theta = rng.random((K, V)) + 1e-8
    theta /= theta.sum(axis=1, keepdims=True)

    return pi, theta

def em_mixture_multinomial(X, K=20, max_iter=50, tol=1e-4, alpha=1e-2, seed=0, vocab=None):
    """
    EM for mixture of multinomials over documents.
    X: csr_matrix (D x V) of counts
    K: number of topics
    alpha: Dirichlet prior (smoothing) on topic-word distributions
    """
    X = X.tocsr()
    D, V = X.shape

    pi, theta = initialize_params(D, V, K, seed=seed)

    log_likelihoods = []
    rng = np.random.default_rng(seed)

    for it in range(max_iter):
        print(f"\n=== EM iteration {it+1}/{max_iter} ===")

        # E-step: responsibilities gamma[d,k] = P(z=k | x_d)
        gamma = np.zeros((D, K), dtype=np.float64)
        log_likelihood = 0.0

        log_theta = np.log(theta + 1e-32)     # K x V
        log_pi = np.log(pi + 1e-32)           # K

        for d in range(D):
            start, end = X.indptr[d], X.indptr[d+1]
            idx = X.indices[start:end]
            cnt = X.data[start:end]

            # log p(x_d | z=k) = sum_w x_dw * log theta_kw
            # shape: (K,)
            if idx.size == 0:
                # empty doc (unlikely, but just in case)
                log_px_z = np.zeros(K)
            else:
                log_px_z = (log_theta[:, idx] * cnt).sum(axis=1)

            log_joint = log_pi + log_px_z  # log p(z, x_d)

            # log-sum-exp for normalization
            m = log_joint.max()
            log_joint_shifted = log_joint - m
            exp_shifted = np.exp(log_joint_shifted)
            denom = exp_shifted.sum()
            gamma[d, :] = exp_shifted / denom

            doc_ll = m + np.log(denom + 1e-32)
            log_likelihood += doc_ll

        log_likelihoods.append(log_likelihood)
        print(f"Log-likelihood: {log_likelihood:.4f}")

        # Check convergence
        if it > 0:
            ll_old = log_likelihoods[-2]
            rel_change = (log_likelihood - ll_old) / (abs(ll_old) + 1e-12)
            print(f"Relative change: {rel_change:.6e}")
            if abs(rel_change) < tol:
                print("Converged based on tolerance.")
                break

        # M-step
        # mixture weights
        Nk = gamma.sum(axis=0)  # shape (K,)
        pi = Nk / D

        # topic-word counts
        print("Updating topic-word distributions...")
        topic_word_counts = np.zeros((K, V), dtype=np.float64)

        for d in range(D):
            start, end = X.indptr[d], X.indptr[d+1]
            idx = X.indices[start:end]
            cnt = X.data[start:end]
            if idx.size == 0:
                continue
            # outer(gamma[d], cnt) -> K x len(idx)
            gw = np.outer(gamma[d], cnt)  # K x L
            # accumulate into columns idx
            topic_word_counts[:, idx] += gw

        # Apply Dirichlet prior alpha
        topic_word_counts += alpha

        # Normalize to get theta
        theta = topic_word_counts / topic_word_counts.sum(axis=1, keepdims=True)

    return pi, theta, np.array(log_likelihoods)

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
    parser.add_argument("--output", type=str, default=f"{DATA_DIR}/em_k20.npz")

    args = parser.parse_args()

    X, vocab = load_data(args.doc_term, args.vocab)

    pi, theta, ll = em_mixture_multinomial(
        X,
        K=args.K,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
        seed=args.seed,
        vocab=vocab
    )

    print("\nFinal log-likelihoods:")
    for i, v in enumerate(ll):
        print(f"Iter {i}: {v:.4f}")

    print("\nTop words per topic:")
    print_top_words(theta, vocab, top_n=10)

    # Save parameters
    out_path = args.output
    # Adjust default name based on K
    if out_path.endswith("em_k20.npz"):
        out_path = f"{DATA_DIR}/em_k{args.K}.npz"

    print(f"\nSaving parameters to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, pi=pi, theta=theta, ll=ll)

    print("Done.")

if __name__ == "__main__":
    main()
