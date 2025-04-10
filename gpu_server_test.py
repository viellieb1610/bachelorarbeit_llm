from classifier import *

df = pd.read_csv("data/yelp_labeled_2k.csv", nrows=100)

clf = OllamaTextClassifier("gemma3:4b")
prompt = load_prompt("prompts/v0.txt")

df["prediction"] = clf.classify(df, prompt)

print(measure_accuracy(df["has_references"], df["prediction"]))