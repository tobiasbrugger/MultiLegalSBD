import glob
import os
import time

import numpy as np
import pandas as pd

import timing
from evaluation import Evaluation
from jsonToDF import jsonToDF
from models import module
from trainer import trainer
from util import Util

columns_eval = ['Model', 'Type', 'Round', 'Precision', 'Recall', 'F1',
                'Mcc', 'Balanced_acc', 'Accuracy', 'Fowlkes_mallows', 'Jaccard']
eval_scores = ['Precision', 'Recall', 'F1',
               'Mcc', 'Balanced_acc', 'Accuracy', 'Fowlkes_mallows', 'Jaccard']


class Experiment():

    util = None
    models = []
    trainers = []
    languages = ['fr', 'it', 'es', 'pt', 'en', 'de']
    types = ['laws', 'judgements', 'both']

    def __init__(self, parameters: dict, stats: bool = False) -> None:
        """
        Loads an experiment
        :param dict parameters: With the following key:value pairs
            :param str : The language to be evaluated
            :param str model_lang: The language the model was trained on
            :param str file_type: The type of files to be evaluated
            :param str model_type: The type of file the model was trained on
        :param bool stats: Wheter to print out stats about the datasets
        """
        self.util = Util()

        self.file_lang = parameters["file_lang"]
        self.model_lang = parameters["model_lang"]
        self.file_type = parameters["file_type"]
        self.model_type = parameters["model_type"]

        
        assert self.model_type in self.types
        assert self.file_type in self.types
        for lang in self.model_lang:
            assert lang in self.languages
        for lang in self.file_lang:
            assert lang in self.languages
        if stats:
            self.stats()

    def load_models(self, models='all') -> None:
        """
        Load given models
        :param models: list of models to load, loads all models by default
        :type models: list[str] or str
        """
        self.models = []
        if models == 'all':
            self.models = [cls(self.model_lang, self.model_type)
                           for cls in module.__subclasses__()]
        elif isinstance(models, list):
            for model in models:
                [self.models.append(
                    cls(self.model_lang, self.model_type)) for cls in module.__subclasses__() if cls.name == model]
        else:
            print('Input modelnames as list or select all by leaving it empty')
        for file_lang in self.file_lang:
            [os.makedirs('data/' + file_lang + '/'+cls.name, exist_ok=True)
            for cls in self.models]

    def load_trainers(self, trainers='all'):
        self.trainers = []
        if isinstance(trainers, list):
            for t in trainers:
                [self.trainers.append(
                    cls(self.model_lang, self.model_type)) for cls in trainer.__subclasses__() if cls.name == t]
        else:
            self.trainers = [cls(self.model_lang, self.model_type)
                             for cls in trainer.__subclasses__()]
        for t in self.trainers:
            t.train_text()

    def train(self):
        """
        Train given trainers
        :param models: list of models to load, loads all models by default
        :type models: list[str] or str
        """
        for t in self.trainers:
            print(f'training: {t.name}')
            t.finalize_training()

    def stats(self):
        """
        Collects the amount of sentences, tokens and charactes for each language
        """
        languages = ['fr', 'it', 'es', 'en', 'de', 'pt']
        columns = ['Language', 'Type', 'Sentences', 'Tokens']
        df = pd.DataFrame(columns=columns)
        total_sentences = 0
        total_tokens = 0
        total_chars = 0
        for lang in languages:

            gold_files = glob.glob('data/'+lang+'/gold/*.jsonl')
            for file in gold_files:
                if "_train" in file or "_test" in file:
                    continue

                sentences, tokens, chars = self.util.stats(file)
                total_sentences += sentences
                total_tokens += tokens
                total_chars += chars
                temp = pd.DataFrame.from_dict({
                    'Language': [lang],
                    'Type': [os.path.basename(file)],
                    'Sentences': [sentences],
                    'Tokens': [tokens]
                })
                df = pd.concat([df, temp], ignore_index=True)
        print((total_sentences, total_tokens))
        df['Type'] = df['Type'].apply(lambda entry: self.util.concat(entry))
        df['Language'] = df['Language'].apply(
            lambda entry: self.util.decode_lang(entry))

        agg_func = {'Sentences': 'sum', 'Tokens': 'sum'}
        df = df.groupby(['Language', 'Type']).agg(agg_func)
        print(df)
        s = df.style.to_latex(
            caption="Statistics on datasets per language and type",
            clines="all;data",
            label="tab:DatasetStats")
        print(s)

    def run(self, round) -> pd.DataFrame:
        """
        Gather gold files,
        Extract gold labels,
        Annotate text with loaded models,
        Extract predicted labels,
        Evaluate/compare gold labels <> predicted labels,
        :return: Pandas Dataframe with model, gold file and scores for each model and gold file
        """
        eval = pd.DataFrame(columns=columns_eval)

        for model in self.models:
            for file_lang in self.file_lang:
                print(f'Evaluating: {model.name}')
                if self.file_type == 'laws':
                    file_string = "Code*"
                elif self.file_type == 'judgements':
                    file_string = "CD*"
                else:
                    file_string = "*"
                if file_lang == 'pt':
                    files = glob.glob(
                        'data/'+file_lang+'/gold/'+file_string+'.jsonl')
    
                    gold_files = [file for file in files if not "test" in file or "train" in file] # 
                else:
                    gold_files = glob.glob(
                        'data/'+file_lang+'/gold/'+file_string+'_test.jsonl')

                for gold_file in gold_files:
                    print(f'Evaluating: {gold_file}')
                    name = os.path.basename(gold_file)
                    file_name, file_extension = os.path.splitext(name)
                    file_name = file_name.removesuffix("_test")
                    df_true = jsonToDF(gold_file)
                    df_gold = self.util.get_gold_labels(df_true)
                    if not os.path.exists('data/'+file_lang+'/'+model.name+'/'+name):
                        df_pred = self.util.annotate(df_true, model)
                        df_pred.to_json('data/'+file_lang+'/' + model.name + '/' + name+'.jsonl', lines=True, force_ascii=False, orient='records')
                    else:
                        df_pred = jsonToDF('data/'+file_lang +
                                        '/' + model.name + '/' + name)
                    evaluation = Evaluation(df_gold, df_pred)
                    data = evaluation.evaluate()

                    entry = pd.DataFrame.from_dict({
                        'Language': [file_lang],
                        'Model': [model.name.removeprefix("mod_")],
                        'Type': [self.concat(file_name)],
                        'Round': round,
                        'Precision': [data['precision']],
                        'Recall': [data['recall']],
                        'F1': [data['f1']],
                        'Mcc': [data['mcc']],
                        'Balanced_acc': [data['balanced_acc']],
                        'Accuracy': [data['accuracy']],
                        'Fowlkes_mallows': [data['fowlkes_mallows']],
                        'Jaccard': [data['jaccard']]
                    })

                    eval = pd.concat([eval, entry], ignore_index=True)

        return eval

    def concat(self, entry):
        return self.util.concat(entry)

    def reset(self):
        """
        Reset annotation-files for all loaded models
        """
        for file_lang in self.file_lang:
            for model in self.models:

                [os.remove(file) for file in glob.glob(
                    'data/'+file_lang+'/'+model.name+'/*.jsonl')]
        print("Reset all model annotations")


if __name__ == '__main__':
    
    parameters = {
        "file_lang": ['de'],
        #['fr', 'es', 'it', 'en', 'de']
        "model_lang": ['de'],
        "file_type": 'both',
        "model_type": 'both'
    }

    experiment = Experiment(
        parameters=parameters,
        stats=False
    )

    baseline = ['nltk', 'nltk_train', 'spacy', 'stanza']
    trained = ['transformer', 'crf', 'bilstm_crf']

    trainers = ['transformer']
    models = ['crf']

    result = pd.DataFrame(columns=columns_eval)
    rounds = 1
    #experiment.load_trainers(trainers=trainers)

    for round in range(rounds):
        print(round)
        #experiment.train()

        experiment.load_models(models=models)

        experiment.reset()
        eval = experiment.run(round)
        result = pd.concat([result, eval], ignore_index=True)

    print(result)
    path = 'data/result_check_' + '_'.join(parameters['file_lang'])+'_'+\
        '_'.join(models) + '_' + "_".join(parameters["model_lang"]) + \
        '_'+parameters["model_type"]+'__' + \
        parameters["file_type"]+'_R'+str(rounds)+'.csv'
    #result.to_csv(path, index=False)

