from classifier import *

df = pd.read_csv("data/imdb_labeled_2k.csv")

clf = OllamaTextClassifier("gemma3:27b")
prompt = "imdb/v7.txt"

df["prediction"] = clf.classify_benchmark(df, prompt, "imdb")

print(measure_accuracy(df["has_references"], df["prediction"]))