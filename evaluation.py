import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             fowlkes_mallows_score, jaccard_score,
                             matthews_corrcoef)
from sklearn.metrics import precision_recall_fscore_support as f1_score


def compare(df1: pd.DataFrame, df2: pd.DataFrame, evaluation: str = "boundary", leniency="lenient"):
    """
    Compares two annotated texts
    filePath1 contains gold-standard annotations
    filePath2 the file that should be compared to it
    evaluation: compares each boundary or segments, defaults to boundary

    Evaluation is done by comparing sentence boundaries
    true positive: predicted boundary is correct
    false positive: predicted boundary is false
    true negative: not occuring
    false negative: not predicted boundary is false

    returns f1-score as float
    """
    assert evaluation == "boundary" or evaluation == "segment", "evaluation has to be: boundary or segment"
    assert leniency == "lenient" or leniency == "strict", "leniecy has to be: lenient or strict"

    true_bounds = []
    pred_bounds = []

    for index, row in df1.iterrows():
        true_bounds.append(row["spans"])
    for index, row in df2.iterrows():
        pred_bounds.append(row["spans"])

    assert len(true_bounds) == len(pred_bounds)

    true_set = set()
    pred_set = set()
    TP = 0
    FP = 0
    FN = 0

    length = len(true_bounds)
    for i in range(length):
        if evaluation == "boundary":
            for sen in true_bounds[i]:
                true_set.add(sen["start"])
                true_set.add(sen["end"])

            for sen in pred_bounds[i]:
                pred_set.add(sen["start"])
                pred_set.add(sen["end"])

        elif evaluation == "segment":
            for sen in true_bounds[i]:
                true_set.add((sen["start"], sen["end"]))

            for sen in pred_bounds[i]:
                pred_set.add((sen["start"], sen["end"]))

        # TODO: Implement lenient scoring
        """
        pred - true  = fp
        
        for bound in fp:
            get two closest bounds in true set -> left, right
            if alphanumeric char in distance bound -> left and bound right
            -> break
            else: add left or right to pred_set
        """

        # true positive
        TP += len(true_set.intersection(pred_set))
        # false positive
        FP += len(pred_set.difference(true_set))
        # false negative
        FN += len(true_set.difference(pred_set))
        # true negative: TODO disregard as it is irrelevant for f1-score?
        true_set.clear()
        pred_set.clear()

    precision = TP/(TP+FP) if (TP+FP) > 0 else 0
    recall = TP/(TP+FN) if (TP+FN) > 0 else 0
    F1 = 2*((precision*recall)/(precision+recall)
            ) if (precision+recall) > 0 else 0

    eval = {
        "f1": F1,
        "precision": precision,
        "recall": recall,
        "tp": TP,
        "fp": FP,
        "fn": FN
    }

    return eval


class Evaluation():
    """
    Evaluates predicted and gold labels
    :param pd.DataFrame df_true: DataFrame containing gold tokens and labels
    :param pd.DataFrame df_pred: DataFrame containing predicted tokens and labels
    """
    y_true = []
    y_pred = []

    def __init__(self, df_true, df_pred) -> None:
        self.y_true = self.getLabels(df_true)
        self.y_pred = self.getLabels(df_pred)
        assert len(self.y_true) == len(self.y_pred)

    def getLabels(self, df:pd.DataFrame) -> list[int]:
        """
        Collects all labels in a DataFrame into a list
        :param pd.DataFrame df: The DataFrame to collect from
        :return: List of all labels
        :rtype: list[int]
        """
        y = []
        for index, row in df.iterrows():
            labels = row["labels"]
            y.extend(labels)
        return y

    def evaluate(self) -> dict:
        """
        Evaluates the given DataFrames
        :return: Dictionary containing evaluation scores
        """
        precision, recall, f1, support = f1_score(
            self.y_true, self.y_pred, average='binary')
        matthews = matthews_corrcoef(self.y_true, self.y_pred)
        accuracy = accuracy_score(self.y_true, self.y_pred)
        balanced_accuracy = balanced_accuracy_score(self.y_true, self.y_pred)
        fowlkes_mallows = fowlkes_mallows_score(self.y_true, self.y_pred)
        jaccard = jaccard_score(self.y_true, self.y_pred)
        
        data = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": matthews,
            "balanced_acc": balanced_accuracy,
            "accuracy": accuracy,
            "fowlkes_mallows": fowlkes_mallows,
            "jaccard": jaccard
        }
        data.update((score, round(value*100, 2))
                    for score, value in data.items())
        #print(data)
        return data
