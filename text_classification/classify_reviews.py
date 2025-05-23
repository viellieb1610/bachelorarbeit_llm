from classifier import *
from tqdm import tqdm

# CONFIG
prompt = "imdb/v7.txt"
platform = "imdb"
CHUNK_SIZE = 100_000
clf = OllamaTextClassifier("gemma3:27b")

# FILES
df_path = f"/home/pwimmer/data/reviews/{platform}_for_classification.csv"
output_path = f"/home/pwimmer/data/results/{platform}2/classified_reviews"
counter = 0

for chunk in tqdm(pd.read_csv(df_path, chunksize=CHUNK_SIZE, index_col=0), desc="Classifying Reviews"):
    if counter != 0:
        chunk["prediction"] = clf.classify(chunk["review_text"], prompt)

        chunk.to_csv(f"{output_path}_{counter}.csv", index=False)
    counter += 1
