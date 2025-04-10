from ollama import chat
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

class OllamaTextClassifier:
    def __init__(self, model):
        self.model = model

    def classify(self, df: pd.DataFrame, prompt: str):
        """
        Gets a df which has to include a column 'review_text'.

        Returns a series.
        """

        texts = df['review_text'].to_list()

        results = []
        for text in tqdm(texts):
            response = chat(model=self.model, messages=[{"role": "user", "content": prompt + text}])
            response = response['message']['content'].strip()
            if response == "Reference to other reviews":
                results.append(1)
            elif response == "No reference":
                results.append(0)
            else:
                results.append(-1)
        return pd.Series(results)

def measure_accuracy(actual: pd.Series, prediction: pd.Series):
    return classification_report(actual, prediction)

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()