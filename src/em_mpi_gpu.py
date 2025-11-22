import argparse
import os
import time

import numpy as np
import torch
from mpi4py import MPI
from scipy.sparse import load_npz


DATA_DIR = "data/processed"


def load_data(doc_term_path, vocab_path):
    print(f"[load_data] Loading {doc_term_path} ...", flush=True)
    X = load_npz(doc_term_path).tocsr()
    D, V = X.shape
    print(f"[load_data] Matrix shape = {X.shape}", flush=True)

    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f]
    assert len(vocab) == V, "Vocab size mismatch"
    return X, vocab


def compute_doc_lengths(X):
    indptr = X.indptr
    return indptr[1:] - indptr[:-1]  # length D


def make_partitions_naive(D, size):
    """Equal number of docs per rank."""
    base = D // size
    rem = D % size
    parts = []
    start = 0
    for r in range(size):
        extra = 1 if r < rem else 0
        end = start + base + extra
        parts.append((start, end))
        start = end
    return parts


def make_partitions_balanced(nnz_per_doc, size):
    """Partition docs to balance total nnz (token counts) per rank."""
    D = len(nnz_per_doc)
    total = int(nnz_per_doc.sum())
    target = total / float(size)

    parts = []
    start = 0
    acc = 0.0
    r = 0

    for d in range(D):
        acc += nnz_per_doc[d]
        if acc >= target and r < size - 1:
            parts.append((start, d + 1))
            start = d + 1
            acc = 0.0
            r += 1

    parts.append((start, D))
    return parts


def get_local_device():
    """Pick GPU based on local MPI rank on the node."""
    # Works with OpenMPI + Slurm
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        print("[warning] CUDA not available, falling back to CPU.", flush=True)
        return torch.device("cpu")
    num_gpus = torch.cuda.device_count()
    device_id = local_rank % max(1, num_gpus)
    device = torch.device(f"cuda:{device_id}")
    print(f"[rank local={local_rank}] Using device {device}", flush=True)
    return device


def em_mixture_multinomial_mpi_gpu(
    X,
    K=20,
    max_iter=30,
    tol=1e-4,
    alpha=1e-2,
    seed=0,
    partition_strategy="balanced",
    comm=MPI.COMM_WORLD,
):
    """
    MPI + GPU EM for mixture of multinomials.

    X: full csr_matrix on each rank (we slice locally)
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    D, V = X.shape
    nnz_per_doc = compute_doc_lengths(X)

    if partition_strategy == "naive":
        parts = make_partitions_naive(D, size)
    elif partition_strategy == "balanced":
        parts = make_partitions_balanced(nnz_per_doc, size)
    else:
        raise ValueError(f"Unknown partition_strategy={partition_strategy}")

    my_start, my_end = parts[rank]
    X_local = X[my_start:my_end].tocsr()
    D_local = X_local.shape[0]
    print(
        f"[rank {rank}] docs {my_start}:{my_end} (D_local={D_local}, nnz={X_local.nnz})",
        flush=True,
    )

    device = get_local_device()
    torch.manual_seed(seed + rank)

    # Precompute CSR arrays for local matrix
    indices = X_local.indices.astype(np.int64)
    data = X_local.data.astype(np.float32)
    indptr = X_local.indptr.astype(np.int64)
    nnz = X_local.nnz

    # Map each nonzero to its document index (0..D_local-1)
    doc_ids = np.empty(nnz, dtype=np.int64)
    for d in range(D_local):
        s, e = indptr[d], indptr[d + 1]
        doc_ids[s:e] = d

    indices_t = torch.from_numpy(indices).to(device)
    data_t = torch.from_numpy(data).to(device)
    indptr_t = torch.from_numpy(indptr).to(device)
    doc_ids_t = torch.from_numpy(doc_ids).to(device)

    # Initialize parameters (same on all ranks)
    torch.manual_seed(seed)
    pi = torch.full((K,), 1.0 / K, device=device)
    theta = torch.rand((K, V), device=device) + 1e-8
    theta = theta / theta.sum(dim=1, keepdim=True)

    # Arrays to track global log-likelihood and timing on rank 0
    ll_hist = []
    iter_times = []
    e_times = []
    m_times = []

    for it in range(max_iter):
        iter_t0 = MPI.Wtime()

        # === E-step (local docs on each rank) ===
        e_t0 = MPI.Wtime()
        gamma = torch.zeros((D_local, K), dtype=torch.float32, device=device)
        log_likelihood_local = 0.0

        log_theta = torch.log(theta + 1e-32)  # (K, V)
        log_pi = torch.log(pi + 1e-32)        # (K,)

        for d in range(D_local):
            start = indptr[d]
            end = indptr[d + 1]
            if start == end:
                gamma[d] = 1.0 / K
                continue

            idx = indices_t[start:end]      # (L,)
            cnt = data_t[start:end]         # (L,)
            log_px_z = (log_theta[:, idx] * cnt.unsqueeze(0)).sum(dim=1)  # (K,)
            log_joint = log_pi + log_px_z

            m = torch.max(log_joint)
            log_joint_shifted = log_joint - m
            exp_shifted = torch.exp(log_joint_shifted)
            denom = exp_shifted.sum()
            gamma_d = exp_shifted / denom

            gamma[d] = gamma_d
            doc_ll = m + torch.log(denom + 1e-32)
            log_likelihood_local += float(doc_ll.item())

        e_t1 = MPI.Wtime()

        # Reduce global log-likelihood
        ll_global = comm.allreduce(log_likelihood_local, op=MPI.SUM)
        if rank == 0:
            ll_hist.append(ll_global)
            print(
                f"\n[EM iter {it+1}/{max_iter}] log-likelihood = {ll_global:.4f}",
                flush=True,
            )

        # Convergence check on rank 0
        converged = False
        if rank == 0 and it > 0:
            ll_old = ll_hist[-2]
            rel_change = (ll_global - ll_old) / (abs(ll_old) + 1e-12)
            print(f"[iter {it+1}] relative change = {rel_change:.6e}", flush=True)
            if abs(rel_change) < tol:
                print("[iter] Converged based on tolerance.", flush=True)
                converged = True
        converged = comm.bcast(converged, root=0)
        if converged:
            e_t1 = MPI.Wtime()  # ensure defined
            m_t1 = MPI.Wtime()
            iter_t1 = MPI.Wtime()
            if rank == 0:
                e_times.append(e_t1 - e_t0)
                m_times.append(m_t1 - e_t1)
                iter_times.append(iter_t1 - iter_t0)
            break

        # === M-step ===
        m_t0 = MPI.Wtime()

        # Local Nk and topic-word counts on GPU
        Nk_local = gamma.sum(dim=0)  # (K,)

        topic_word_counts_local = torch.zeros(
            (K, V), dtype=torch.float32, device=device
        )

        # For each nonzero: add gamma[d,k] * count to topic_word_counts[k, w]
        gamma_n = gamma[doc_ids_t, :]                    # (nnz, K)
        gw = gamma_n * data_t.unsqueeze(1)               # (nnz, K)
        gw_T = gw.t()                                    # (K, nnz)

        for k in range(K):
            topic_word_counts_local[k].index_add_(0, indices_t, gw_T[k])

        # Move to CPU for MPI Allreduce
        Nk_local_np = Nk_local.detach().cpu().numpy().astype(np.float64)
        twc_local_np = topic_word_counts_local.detach().cpu().numpy().astype(np.float64)

        # Global reductions
        Nk_global = np.empty_like(Nk_local_np)
        comm.Allreduce(Nk_local_np, Nk_global, op=MPI.SUM)

        twc_global = np.empty_like(twc_local_np)
        comm.Allreduce(twc_local_np, twc_global, op=MPI.SUM)

        # Convert back to torch on device
        Nk_t = torch.from_numpy(Nk_global).to(device=device, dtype=torch.float32)
        topic_word_counts = torch.from_numpy(twc_global).to(
            device=device, dtype=torch.float32
        )

        # Update pi, theta (same on all ranks)
        pi = Nk_t / float(D)
        topic_word_counts += alpha
        theta = topic_word_counts / topic_word_counts.sum(dim=1, keepdim=True)

        m_t1 = MPI.Wtime()
        iter_t1 = MPI.Wtime()

        if rank == 0:
            e_times.append(e_t1 - e_t0)
            m_times.append(m_t1 - m_t0)
            iter_times.append(iter_t1 - iter_t0)

    # Gather results from rank 0 only
    if rank == 0:
        pi_np = pi.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        ll_np = np.array(ll_hist, dtype=np.float64)
        iter_times_np = np.array(iter_times, dtype=np.float64)
        e_times_np = np.array(e_times, dtype=np.float64)
        m_times_np = np.array(m_times, dtype=np.float64)
    else:
        pi_np = theta_np = ll_np = iter_times_np = e_times_np = m_times_np = None

    return pi_np, theta_np, ll_np, iter_times_np, e_times_np, m_times_np


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_term", type=str, default=f"{DATA_DIR}/doc_term.npz")
    parser.add_argument("--vocab", type=str, default=f"{DATA_DIR}/vocab.txt")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="balanced",
        choices=["naive", "balanced"],
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=f"{DATA_DIR}/em_mpi_gpu",
    )

    args = parser.parse_args()

    if rank == 0:
        X, vocab = load_data(args.doc_term, args.vocab)
    else:
        X = vocab = None

    # Broadcast X and vocab by re-loading on each rank (simpler):
    # all ranks read from shared filesystem using same paths.
    # This keeps code simple; memory scaling discussion will note
    # model replicated on all ranks.
    if rank != 0:
        X, vocab = load_data(args.doc_term, args.vocab)

    pi, theta, ll, iter_times, e_times, m_times = em_mixture_multinomial_mpi_gpu(
        X,
        K=args.K,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
        seed=args.seed,
        partition_strategy=args.partition_strategy,
        comm=comm,
    )

    if rank == 0:
        out_base = f"{args.output_prefix}_K{args.K}_P{comm.Get_size()}_{args.partition_strategy}"
        np.savez(
            out_base + ".npz",
            pi=pi,
            theta=theta,
            ll=ll,
            iter_times=iter_times,
            e_times=e_times,
            m_times=m_times,
        )
        print(f"[rank 0] Saved parameters and timings to {out_base}.npz", flush=True)


if __name__ == "__main__":
    main()
