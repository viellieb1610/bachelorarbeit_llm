from classifier import *

df = pd.read_csv("data/yelp_labeled_2k.csv")

clf = OllamaTextClassifier("gemma3:27b")
prompt = "yelp/v6_1.txt"

df["prediction"] = clf.classify(df, prompt, "yelp")

print(measure_accuracy(df["has_references"], df["prediction"]))