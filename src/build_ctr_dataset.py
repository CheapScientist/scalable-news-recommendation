import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = "../data/mind"
OUT_DIR = "../data/processed"


def main():
    print("Loading behaviors.tsv...")

    behaviors = pd.read_csv(
        f"{DATA_DIR}/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )

    rows = []

    print("Parsing impressions...")
    for idx, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
        user = row["user_id"]
        imp = str(row["impressions"])

        for item in imp.split():
            try:
                news_id, label = item.split("-")
                label = int(label)
                rows.append((user, news_id, label))
            except:
                # Skip malformed entries (rare)
                continue

    print("Building DataFrame...")
    ctr = pd.DataFrame(rows, columns=["user_id", "news_id", "label"])
    print(ctr.head())
    print("Total interactions:", len(ctr))

    print("Saving ctr.csv...")
    os.makedirs(OUT_DIR, exist_ok=True)
    ctr.to_csv(f"{OUT_DIR}/ctr.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
