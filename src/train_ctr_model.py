import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Metrics
# -----------------------------
def ndcg_at_k(y_true, y_score, k=10):
    """
    NDCG@k for binary relevance.
    Handles cases where user has < k impressions.
    """
    if y_true.sum() == 0:
        return None

    # Sort by predicted score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order][:k]
    L = len(y_true_sorted)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    discounts = discounts[:L]   # fix broadcasting

    dcg = (y_true_sorted * discounts).sum()

    num_pos = int(min(y_true.sum(), L))
    ideal = np.ones(num_pos)
    idcg = (ideal * discounts[:num_pos]).sum()

    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k_per_user(user_ids, y_true, y_score, k=10):
    df = pd.DataFrame({
        "user_id": user_ids,
        "y_true": y_true,
        "y_score": y_score
    })

    ndcgs = []
    for user, group in df.groupby("user_id"):
        ndcg = ndcg_at_k(group["y_true"].values,
                         group["y_score"].values,
                         k=k)
        if ndcg is not None:
            ndcgs.append(ndcg)

    return float(np.mean(ndcgs)) if ndcgs else float("nan")


# -----------------------------
# MLP Model
# -----------------------------
class CTRMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_mlp(X_train, y_train, X_val, y_val,
              batch_size=256, epochs=10, lr=1e-3, device="cpu"):
    input_dim = X_train.shape[1]
    model = CTRMLP(input_dim=input_dim, hidden_dim=64).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)          # (B,)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[MLP] Epoch {epoch}/{epochs} - train loss: {avg_loss:.4f}")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().to(device)
        logits = model(X_val_t).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

    return model, probs


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/ctr_features.npy")
    parser.add_argument("--labels", type=str, default="data/processed/ctr_labels.npy")
    parser.add_argument("--ctr_csv", type=str, default="data/processed/ctr.csv",
                        help="Original CTR CSV to get user_id for NDCG grouping")
    parser.add_argument("--model", type=str, default="both",
                        choices=["logreg", "mlp", "both"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="data/processed/")
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    print("[load] Features:", args.features)
    X = np.load(args.features)    # (N, 2K)
    print("[load] Labels:", args.labels)
    y = np.load(args.labels)      # (N,)

    print("[load] CTR CSV for user_ids:", args.ctr_csv)
    ctr_df = pd.read_csv(args.ctr_csv)
    if len(ctr_df) != len(X):
        print(f"[warn] ctr.csv rows ({len(ctr_df)}) != features rows ({len(X)}). "
              "Assuming same order as when features were built and truncating.")
        n = min(len(ctr_df), len(X))
        ctr_df = ctr_df.iloc[:n]
        X = X[:n]
        y = y[:n]

    user_ids = ctr_df["user_id"].to_numpy()

    # Train/val split (row-wise)
    X_train, X_val, y_train, y_val, users_train, users_val = train_test_split(
        X, y, user_ids,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y
    )

    print(f"[info] Train size: {len(X_train)}, Val size: {len(X_val)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    if args.model in ("logreg", "both"):
        print("\n=== Training Logistic Regression ===")
        # C=1.0 default, you can tune this
        logreg = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1
        )
        logreg.fit(X_train, y_train)
        prob_val = logreg.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, prob_val)
        ndcg = ndcg_at_k_per_user(users_val, y_val, prob_val, k=10)
        print(f"[LOGREG] AUC = {auc:.4f}, NDCG@10 = {ndcg:.4f}")

        # save model coefficients for reproducibility
        np.save(os.path.join(args.out_dir, "logreg_coef.npy"), logreg.coef_)
        np.save(os.path.join(args.out_dir, "logreg_intercept.npy"), logreg.intercept_)

    # -----------------------------
    # MLP
    # -----------------------------
    if args.model in ("mlp", "both"):
        print("\n=== Training MLP ===")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MLP] Using device: {device}")

        mlp_model, prob_val = train_mlp(
            X_train, y_train,
            X_val, y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )

        auc = roc_auc_score(y_val, prob_val)
        ndcg = ndcg_at_k_per_user(users_val, y_val, prob_val, k=10)
        print(f"[MLP] AUC = {auc:.4f}, NDCG@10 = {ndcg:.4f}")

        # save model state dict
        torch.save(mlp_model.state_dict(),
                   os.path.join(args.out_dir, "ctr_mlp.pt"))

    print("\n[done] CTR training complete.")


if __name__ == "__main__":
    main()
