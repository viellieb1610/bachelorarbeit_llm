from ollama import chat
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import datetime


class OllamaTextClassifier:
    def __init__(self, model):
        self.model = model

    def classify_benchmark(self, df: pd.DataFrame, prompt: str, platform: str):
        """
        Gets a df which has to include a column 'review_text'.

        Returns a series.
        """

        texts = df['review_text'].to_list()
        loaded_prompt = load_prompt(f"prompts/{prompt}")

        results = []
        answers = []
        # print(f"Classifying {len(texts)} {platform} texts with model {self.model} using prompt: {prompt}")
        for text in tqdm(texts):
            #text = '\nInput:\n"' + text + '"\nOutput:'
            text = f'\nInput:\n"{text}"\nOutput:'
            response = chat(model=self.model, messages=[{"role": "user", "content": loaded_prompt + text}])
            response = response['message']['content'].strip()
            response = " ".join(response.splitlines())
            answers.append(response)
            if "Reference to other reviews" in response:
                results.append(1)
            elif "No reference" in response:
                results.append(0)
            else:
                # print("UNEXPECTED ANSWER:\\"+response+"\n\n")
                results.append(-1)
        df['prediction'] = results
        df['answers'] = answers
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        df.to_csv(f"runs/{platform}/{timestamp}-{self.model}-{len(texts)}.csv")
        return pd.Series(results)

    def classify(self, reviews: pd.Series, prompt: str):
        """
        Takes reviews to classify, classifies them and returns a series with results either 0 or 1.
        :param reviews:
        :param prompt:
        :return: Series with results
        """
        texts = reviews.to_list()
        loaded_prompt = load_prompt(f"prompts/{prompt}")

        results = []
        for text in tqdm(texts):
            text = f'\nInput:\n"{text}"\nOutput:'
            response = chat(model=self.model, messages=[{"role": "user", "content": loaded_prompt + text}])
            response = response['message']['content'].strip()
            response = " ".join(response.splitlines())
            if "Reference to other reviews" in response:
                results.append(1)
            elif "No reference" in response:
                results.append(0)
            else:
                results.append(-1)

        return pd.Series(results)

def measure_accuracy(actual: pd.Series, prediction: pd.Series):
    return classification_report(actual, prediction)

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()