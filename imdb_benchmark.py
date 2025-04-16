from classifier import *

df = pd.read_csv("data/imdb_labeled_2k.csv")

clf = OllamaTextClassifier("llama3.3")
prompt = "imdb/v4.txt"

df["prediction"] = clf.classify(df, prompt, "imdb")

print(measure_accuracy(df["has_references"], df["prediction"]))