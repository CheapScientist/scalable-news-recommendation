import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import os
from tqdm import tqdm
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

DATA_DIR = "../data/mind"
OUT_DIR = "../data/processed"

MAX_VOCAB = 30000
MIN_WORD_FREQ = 5

def clean_text(s):
    if isinstance(s, float):  # handle NaN
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokenize(s):
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def build_vocab(texts):
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)

    # Filter out rare words
    filtered = {w: c for w, c in counter.items() if c >= MIN_WORD_FREQ}

    # Take top MAX_VOCAB
    most_common = sorted(filtered.items(), key=lambda x: -x[1])[:MAX_VOCAB]

    word2id = {w: idx for idx, (w, _) in enumerate(most_common)}

    return word2id

def main():
    print("Loading news.tsv...")
    news = pd.read_csv(
        f"{DATA_DIR}/news.tsv",
        sep="\t",
        header=None,
        names=["news_id", "category", "subcategory",
               "title", "abstract", "url",
               "title_entities", "abstract_entities"]
    )

    print("Cleaning + tokenizing article text...")
    corpus = []
    for idx, row in tqdm(news.iterrows(), total=len(news)):
        text = clean_text(str(row["title"]) + " " + str(row["abstract"]))
        tokens = tokenize(text)
        corpus.append(tokens)

    print("Building vocabulary...")
    word2id = build_vocab(corpus)
    vocab_size = len(word2id)
    print("Vocab size =", vocab_size)

    print("Building CSR document-term matrix...")
    rows = []
    cols = []
    vals = []

    for i, tokens in enumerate(tqdm(corpus)):
        for w in tokens:
            if w in word2id:
                j = word2id[w]
                rows.append(i)
                cols.append(j)
                vals.append(1)

    X = csr_matrix((vals, (rows, cols)), shape=(len(corpus), vocab_size))
    print("Matrix shape:", X.shape)

    print("Saving outputs...")
    os.makedirs(OUT_DIR, exist_ok=True)

    save_npz(f"{OUT_DIR}/doc_term.npz", X)
    news.to_csv(f"{OUT_DIR}/news_meta.csv", index=False)

    with open(f"{OUT_DIR}/vocab.txt", "w") as f:
        for w in word2id:
            f.write(w + "\n")

    print("Done!")

if __name__ == "__main__":
    main()
