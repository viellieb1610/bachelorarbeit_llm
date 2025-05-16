from classifier import *

df = pd.read_csv("../data/yelp_labeled_2k.csv")

clf = OllamaTextClassifier("Gemma3:12b")
prompt = "yelp/v10.txt"

df["prediction"] = clf.classify_benchmark(df, prompt, "yelp")

print(measure_accuracy(df["has_references"], df["prediction"]))