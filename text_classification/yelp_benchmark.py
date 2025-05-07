from classifier import *

df = pd.read_csv("../data/yelp_labeled_2k.csv")

clf = OllamaTextClassifier("gemma3:27b")
prompt = "yelp/v0.txt"

df["prediction"] = clf.classify_benchmark(df, prompt, "yelp")

print(measure_accuracy(df["has_references"], df["prediction"]))