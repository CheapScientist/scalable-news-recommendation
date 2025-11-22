import numpy as np
import pandas as pd
import argparse
import os


def normalize_rows(mat):
    """Row-normalize a 2D numpy array."""
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-12
    return mat / row_sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gamma_file",
        type=str,
        default="../data/processed/gamma_K20.npy",
        help="Numpy file with gamma (D x K) document-topic responsibilities",
    )
    parser.add_argument(
        "--news_meta",
        type=str,
        default="../data/processed/news_meta.csv",
        help="Metadata CSV with news_id column aligned to doc_term rows",
    )
    parser.add_argument(
        "--ctr_file",
        type=str,
        default="../data/processed/ctr.csv",
        help="CTR dataset with user_id, news_id, label",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../data/processed/",
        help="Output directory for topic feature files",
    )
    args = parser.parse_args()

    # -----------------------------
    # 1. Load gamma and news_meta
    # -----------------------------
    print("[load] gamma from:", args.gamma_file)
    gamma = np.load(args.gamma_file)   # shape (D, K)
    D, K = gamma.shape
    print(f"[info] gamma shape = ({D}, {K})")

    print("[load] news_meta from:", args.news_meta)
    news_meta = pd.read_csv(args.news_meta)

    if len(news_meta) != D:
        print(
            f"[warn] news_meta rows ({len(news_meta)}) != gamma docs ({D}). "
            "Assuming first D rows align with doc_term."
        )

    # news_id -> row index mapping
    news_to_idx = {nid: i for i, nid in enumerate(news_meta["news_id"])}
    print("[info] news_to_idx size:", len(news_to_idx))

    # -----------------------------
    # 2. Build Document-Topic Matrix
    # -----------------------------
    print("[step] Normalizing gamma to get doc-topic vectors...")
    doc_topics = normalize_rows(gamma)   # (D, K)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    doc_out = os.path.join(out_dir, "doc_topics.npy")
    np.save(doc_out, doc_topics)
    print("[save] Document-topic vectors ->", doc_out)

    # -----------------------------
    # 3. Build User-Topic Matrix
    # -----------------------------
    print("[load] CTR file from:", args.ctr_file)
    ctr = pd.read_csv(args.ctr_file)

    users = sorted(ctr["user_id"].unique())
    user_topics = {}

    print("[step] Accumulating user-topic vectors (vectorized)...")

    # 1) Keep only positive clicks
    clicks = ctr[ctr["label"] == 1].copy()

    # 2) Map news_id -> doc index once
    clicks["doc_idx"] = clicks["news_id"].map(news_to_idx)

    # 3) Drop rows where we don't have a doc index
    clicks = clicks.dropna(subset=["doc_idx"])
    clicks["doc_idx"] = clicks["doc_idx"].astype(int)

    # 4) Group by user and average doc topic vectors
    click_groups = clicks.groupby("user_id")["doc_idx"]

    for i, (user, idx_series) in enumerate(click_groups):
        idxs = idx_series.to_numpy()
        valid = idxs[(idxs >= 0) & (idxs < D)]
        if len(valid) == 0:
            user_topics[user] = np.zeros(K, dtype=np.float32)
        else:
            user_topics[user] = doc_topics[valid].mean(axis=0).astype(np.float32)

        # Optional progress print
        if (i + 1) % 1000 == 0:
            print(f"[progress] processed {i + 1} users with clicks")

    # 5) Handle users with no positive clicks (cold-start)
    for user in users:
        if user not in user_topics:
            user_topics[user] = np.zeros(K, dtype=np.float32)

    U = np.vstack([user_topics[u] for u in users])

    user_out = os.path.join(out_dir, "user_topics.npy")
    np.save(user_out, U)
    print("[save] User-topic vectors ->", user_out)

    # -----------------------------
    # 4. Build CTR Training Features
    # -----------------------------
    print("[step] Building CTR features (vectorized)...")

    # Map news_id -> doc index
    ctr["doc_idx"] = ctr["news_id"].map(news_to_idx).astype(float)

    # Drop rows where doc_idx is missing
    ctr = ctr.dropna(subset=["doc_idx"])
    ctr["doc_idx"] = ctr["doc_idx"].astype(int)

    # Map user_id -> user index
    user_to_idx = {u: i for i, u in enumerate(users)}
    ctr["user_idx"] = ctr["user_id"].map(user_to_idx)

    # Convert to numpy arrays
    doc_idx = ctr["doc_idx"].to_numpy()
    user_idx = ctr["user_idx"].to_numpy()
    labels = ctr["label"].to_numpy()

    # Gather vectors in one shot
    user_vecs = U[user_idx]  # (N, K)
    doc_vecs = doc_topics[doc_idx]  # (N, K)

    # Concatenate into features
    X = np.hstack([user_vecs, doc_vecs]).astype(np.float32)
    y = labels.astype(np.int64)

    # Save
    X_out = os.path.join(out_dir, "ctr_features.npy")
    y_out = os.path.join(out_dir, "ctr_labels.npy")
    np.save(X_out, X)
    np.save(y_out, y)

    print("[save] CTR features ->", X_out)
    print("[save] CTR labels  ->", y_out)
    print("[done] Vectorized CTR feature construction complete.")


if __name__ == "__main__":
    main()
