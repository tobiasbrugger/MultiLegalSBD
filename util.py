from itertools import repeat

import pandas as pd

from jsonToDF import jsonToDF
from models import *
from tokenizer import sentences2tokens, sentences2tokensWithWS


class Util():

    def decode_lang(self, lang: str) -> str:
        """
        Decodes language short form to normal form
        :param str lang: Language short form
        :return: Normal form
        :rtype: str
        """
        match lang:
            case "fr":
                return "French"
            case "it":
                return "Italian"
            case "es":
                return "Spanish"
            case "en":
                return "English"
            case "de":
                return "German"
            case "pt":
                return "Portuguese"
            case _:
                return "Unknown"

    def decode_type(self, type: str) -> str:
        match type:
            case "both":
                return "Judgements + Laws"
            case "laws":
                return "Laws"
            case "judgements":
                return "Judgements"
            case _:
                return "Unknown"

    def concat(self, entry):
        if "CD" in entry:
            return "Judgements"
        elif "Code" in entry:
            return "Laws"
        elif "Constitution" in entry:
            return "Laws"
        elif "agb" in entry:
            return "ToS"
        else:
            return entry

    def add_value_labels(self, ax, vspacing=-25, hspacing=0):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            vspace = vspacing
            hspace = hspacing
            # Vertical alignment for positive values
            va = 'center'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.2f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                 # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                # Vertically shift label by `space`
                xytext=(hspace, vspace),
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,
                fontsize=14)                      # Vertically align label differently for
            # positive and negative values.

    def stats(self, filePath: str):
        """
        Collects stats about text in a given file
        :param str filePath: Path to file
        :return: Amount of sentences, tokens, characters
        :rtype: tuple(int,int,int)
        """
        df = jsonToDF(filePath)
        total_tokens = 0
        total_sentences = 0
        total_chars = 0
        for index, row in df.iterrows():
            sentences = row["spans"]
            total_sentences += len(sentences)
            tokens = row['tokens']
            total_tokens += len(tokens)
            text = row['text']
            total_chars += len(text)
        return total_sentences, total_tokens, total_chars

    def get_gold_labels(self, df_true: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gold tokens and labels from a gold file
        :param pd.DataFrame df_true: DataFrame containing gold-annotations
        :return: DataFrame containing gold tokens and labels
        :rtype: pd.DataFrame
        """
        jsonObject = []
        for index, row in df_true.iterrows():
            text = row["text"]
            spans = row['spans']
            #tokens = [token['text'] for token in row['tokens']]
            tokens = [token.group() for token in sentences2tokens(text)]
            labels = list(repeat(0, len(tokens)))
            for span in spans:
                subtext = text[span['start']:span['end']]
                start = span['token_start']
                end = span['token_end']
                """for i in range(start, end):
                    labels[i] = 3"""
                labels[start] = 1
                labels[end] = 1

            jsonObject.append({
                "tokens": tokens,
                "labels": labels
            })

        return pd.DataFrame.from_dict(jsonObject)

    def annotate(self, df: pd.DataFrame, model: module) -> pd.DataFrame:
        """
        Predict Sentence Boundaries for text in a Pandas Dataframe using the given model
        :param DataFrame df: DataFrame containing columns: text, spans
        :param module model: The model to predict Sentence Boundaries
        :return: DataFrame containing tokens and labels
        :rtype: pd.DataFrame
        """
        jsonObject = []

        for index, row in df.iterrows():
            #print("annotating: ", index)
            text = row["text"]
            spans = row['spans']
            if model.name == 'transformer' or model.name == 'bilstm_crf':
                # list of tuple(token, label)
                sentences = model.sentenize(text, spans)
            else:
                # list of tuple(token, label)
                sentences = model.sentenize(text)
            tokens = [tokens for tokens, labels in sentences]
            labels = [labels for tokens, labels in sentences]

            jsonObject.append({
                "tokens": tokens,
                "labels": labels
            })
        return pd.DataFrame.from_dict(jsonObject)
