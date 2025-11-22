import numpy as np
from scipy.sparse import load_npz, save_npz
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/processed/doc_term.npz")
    parser.add_argument("--output_dir", type=str, default="../data/processed/")
    args = parser.parse_args()

    X = load_npz(args.input)
    D_total = X.shape[0]

    # sizes for weak scaling
    sizes = {
        "13k": 13280,        # for 1 GPU
        "26k": 26560,        # for 2 GPUs
    }

    for name, n_docs in sizes.items():
        n = min(n_docs, D_total)
        X_sub = X[:n]
        out_path = os.path.join(args.output_dir, f"doc_term_{name}.npz")
        save_npz(out_path, X_sub)
        print(f"Saved subset: {out_path} (shape={X_sub.shape})")

if __name__ == "__main__":
    main()
