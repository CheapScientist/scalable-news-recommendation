import pandas as pd

news_path = "../data/mind/news.tsv"
beh_path = "../data/mind/behaviors.tsv"

news = pd.read_csv(news_path,
                   sep='\t',
                   header=None,
                   names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])

print("NEWS SHAPE:", news.shape)
print(news.head())

behaviors = pd.read_csv(beh_path,
                        sep='\t',
                        header=None,
                        names=["impression_id", "user_id", "time", "history", "impressions"])

print("\nBEHAVIORS SHAPE:", behaviors.shape)
print(behaviors.head())
