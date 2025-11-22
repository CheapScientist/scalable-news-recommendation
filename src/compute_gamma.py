import numpy as np
from scipy.sparse import load_npz
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--em_file",
        type=str,
        default="../data/processed/em_mpi_gpu_K20_P4_balanced.npz",
        help="NPZ file containing pi and theta",
    )
    parser.add_argument(
        "--doc_term",
        type=str,
        default="../data/processed/doc_term.npz",
        help="CSR document-term matrix",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../data/processed/gamma_K20.npy",
        help="Output path for gamma (D x K) numpy array",
    )
    args = parser.parse_args()

    print("[load] EM params from:", args.em_file)
    em = np.load(args.em_file)
    pi = em["pi"]          # shape (K,)
    theta = em["theta"]    # shape (K, V)

    K, V = theta.shape
    print(f"[info] theta shape = ({K}, {V})")

    print("[load] doc-term matrix from:", args.doc_term)
    X = load_npz(args.doc_term).tocsr()
    D, Vx = X.shape
    print(f"[info] X shape = ({D}, {Vx})")

    if Vx != V:
        raise ValueError(f"Vocab size mismatch: theta has V={V}, X has V={Vx}")

    # Precompute logs for mixture-of-multinomials
    eps = 1e-12
    log_theta = np.log(theta + eps)       # (K, V)
    log_pi = np.log(pi + eps)            # (K,)

    gamma = np.empty((D, K), dtype=np.float32)

    print("[step] Computing gamma for each document...")
    indptr = X.indptr
    indices = X.indices
    data = X.data

    for d in range(D):
        start = indptr[d]
        end = indptr[d + 1]
        idx = indices[start:end]   # word indices
        cnt = data[start:end]      # counts

        # log p(x_d | z=k) ∝ sum_w x_dw * log theta_kw
        log_joint = log_pi.copy()

        if len(idx) > 0:
            # log_theta[:, idx] -> (K, nnz_d), cnt -> (nnz_d,)
            # broadcasting: (K, nnz_d) * (nnz_d,) -> (K, nnz_d)
            log_joint += (log_theta[:, idx] * cnt).sum(axis=1)

        # Stabilize and normalize
        m = log_joint.max()
        exps = np.exp(log_joint - m)
        gamma[d, :] = exps / (exps.sum() + eps)

        if (d + 1) % 5000 == 0:
            print(f"[progress] processed {d+1}/{D} docs")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, gamma)
    print("[save] gamma saved to:", out_path)
    print("[done] gamma recomputation complete.")


if __name__ == "__main__":
    main()
