import glob
import json
import os
import pickle
import random
import subprocess
import sys
from itertools import repeat

import evaluate
import nltk.tokenize.punkt as punkt
import numpy as np
import pandas as pd
import pycrfsuite
import torch
from accelerate import Accelerator
from datasets import (ClassLabel, Dataset, DatasetDict, Features, Sequence,
                      Value, concatenate_datasets, load_dataset,
                      load_from_disk)
from huggingface_hub import Repository, get_full_repo_name
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, get_scheduler, set_seed)

from crf_features import CRF_Features
from jsonToDF import jsonToDF
from tokenizer import sentences2tokens, sentences2tokensWithWS


class trainer(object):
    """
    Base class for trainers
    """

    def __init__(self, lang: list) -> None:
        """
        :param list lang: List of languages the trainer is initialized for
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train_text(self):
        """
        Creates/gathers text to train on
        """
        raise NotImplementedError

    def finalize_training(self):
        """
        finalizes the training process
        """
        raise NotImplementedError


class trainer_nltk(trainer):

    name = 'nltk_train'
    model_path = ''
    punkt_trainer = None
    model = None

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang
        self.model_path = 'models/nltk_model_' + \
            '_'.join(self.lang)+'_trained.mdl'

    def train_text(self):
        files = []
        for l in self.lang:
            files.extend(glob.glob('data/'+l+'/gold/*.jsonl'))

        for file in files:
            if "_train" in file:
                continue
            if "_test" in file:
                continue
            df = jsonToDF(file)
            for index, row in df.iterrows():
                text = row["text"]
                self.train(text)

    def finalize_training(self):
        self.punkt_trainer.finalize_training()
        self.model = PunktSentenceTokenizer(self.punkt_trainer.get_params())
        with open(self.model_path, mode='wb') as out:
            pickle.dump(self.model, out, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, text):
        if not self.punkt_trainer:
            self.punkt_trainer = punkt.PunktTrainer()
        self.punkt_trainer.train(text, finalize=False)


class trainer_crf(trainer):

    name = 'crf'
    train_sentences = []
    test_sentences = []
    features = None
    trainer = None
    model_path = ''

    def __init__(self, lang: list, model_type='both') -> None:
        self.lang_string = "_".join(lang)
        self.lang = lang
        self.model_type = model_type
        self.features = CRF_Features()
        self.iterations = 100
        self.c1 = 1
        self.c2 = 1e-3
        if self.model_type == 'laws':
            self.model_path = 'models/crf_'+self.lang_string+'_' + self.model_type+'.crfsuite'
        elif self.model_type == 'judgements':
            self.model_path = 'models/crf_'+self.lang_string+'_' + self.model_type+'.crfsuite'
        else:
            self.model_path = 'models/crf_'+self.lang_string+'.crfsuite'

    def train_text(self):
        self.train_sentences = []
        self.test_sentences = []
        if self.model_type == 'laws':
            train_string = '*Code_train'
            test_string = '*Code_test'
        elif self.model_type == 'judgements':
            train_string = 'CD*_train'
            test_string = 'CD*_test'
        else:
            train_string = '*_train'
            test_string = '*_test'
        self.train_files = []
        self.test_files = []
        for lang in self.lang:
            files = glob.glob('data/'+lang+'/gold/'+train_string+'.jsonl')
            self.train_files.extend(files)
            files = glob.glob('data/'+lang+'/gold/'+test_string+'.jsonl')
            self.test_files.extend(files)

        for file in self.train_files:
            if "Constitution" in file:
                continue
            print(file)
            df = jsonToDF(file)
            for index, row in df.iterrows():
                text = row['text']
                spans = row['spans']
                tokenList = sentences2tokensWithWS(text)
                tokens = [token.group() for token in tokenList]
                labels = list(repeat('O', len(tokenList)))
                for span in spans:
                    start = span["start"]
                    end = span["end"]
                    subtokens = []
                    for i, token in enumerate(tokenList):
                        if token.end() > end:
                            break
                        if token.start() < start:
                            continue
                        if token.start() >= start and token.end() <= end:
                            subtokens.append((i, token))
                    for i, token in enumerate(subtokens):
                        index = token[0]
                        if i == 0:
                            labels[index] = 'B-Sentence'
                        elif i == len(subtokens)-1:
                            labels[index] = 'L-Sentence'
                        else:
                            labels[index] = 'I-Sentence'

                self.train_sentences.append(list(zip(tokens, labels)))

        for file in self.test_files:
            if "Constitution" in file:
                continue

            df = jsonToDF(file)
            for index, row in df.iterrows():
                text = row['text']
                spans = row['spans']
                tokenList = sentences2tokensWithWS(text)
                tokens = [token.group() for token in tokenList]
                labels = list(repeat('O', len(tokenList)))
                for span in spans:
                    start = span["start"]
                    end = span["end"]
                    subtokens = []
                    for i, token in enumerate(tokenList):
                        if token.end() > end:
                            break
                        if token.start() < start:
                            continue
                        if token.start() >= start and token.end() <= end:
                            subtokens.append((i, token))
                    for i, token in enumerate(subtokens):
                        index = token[0]
                        if i == 0:
                            labels[index] = 'B-Sentence'
                        elif i == len(subtokens)-1:
                            labels[index] = 'L-Sentence'
                        else:
                            labels[index] = 'I-Sentence'

                self.test_sentences.append(list(zip(tokens, labels)))

    def get_data(self):
        for sent in self.train_sentences:
            yield self.features.generate_features(sent)

    def finalize_training(self):
        print("starting training")
        """x_train = [self.features.generate_features(
            sent) for sent in self.train_sentences]"""

        y_train = [self.features.sent2labels(
            sent) for sent in self.train_sentences]

        self.trainer = pycrfsuite.Trainer(verbose=True)

        for xseq, yseq in zip(self.get_data(), y_train):
            self.trainer.append(xseq, yseq)

        x_train = []
        y_train = []
        self.trainer.set_params({
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.iterations,
            'feature.possible_transitions': True
        })
        self.trainer.train(self.model_path)

        x_test = [self.features.generate_features(
            sent) for sent in self.test_sentences]
        y_test = [self.features.sent2labels(sent)
                  for sent in self.test_sentences]

        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_path)
        y_pred = [tagger.tag(xseq) for xseq in x_test]
        print(self.features.bio_classification_report(y_test, y_pred))

    def train(self):
        pass


class trainer_bilstm_crf(trainer):

    name = 'bilstm_crf'

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang
        self.model_type = model_type
        if self.model_type == 'judgements':
            self.base_path = "utility/bilstm_corpus_" + \
                "_".join(self.lang)+'_'+self.model_type
            self.model_dir = "models/bilstm_crf_" + \
                "_".join(self.lang)+'_'+self.model_type
        elif self.model_type == 'laws':
            self.base_path = "utility/bilstm_corpus_" + \
                "_".join(self.lang)+'_'+self.model_type
            self.model_dir = "models/bilstm_crf_" + \
                "_".join(self.lang)+'_'+self.model_type
        else:
            self.base_path = "utility/bilstm_corpus_"+"_".join(self.lang)
            self.model_dir = "models/bilstm_crf_" + "_".join(self.lang)

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.max_seq_len = "512"
        self.device = "cuda:0"
        self.embedding_dim = "128"
        self.hidden_dim = "256"
        self.num_epoch = "8"
        self.batch_size = "16"  # 16
        self.lr = "0.01"
        self.weight_decay = "0.0001"
        self.recovery = False
        self.save_best_val_model = True
        self.val_split = "0.2"
        self.test_split = "0"

    def train_text(self):
        for f in os.listdir(self.base_path):
            os.remove(os.path.join(self.base_path, f))
        for f in os.listdir(self.model_dir):
            os.remove(os.path.join(self.model_dir, f))

        files = []

        if self.model_type == "judgements":
            file_string = "CD*"
        elif self.model_type == "laws":
            file_string = "*Code"
        else:
            file_string = "*"
        for lang in self.lang:
            files.extend(glob.glob('data/'+lang+'/gold/' +
                         file_string+'_train.jsonl'))

        tokenlist = []
        labellist = []

        for file in files:
            if "Constitution" in file:
                continue
            if "_test" in file:
                continue
            if "agb" in file:
                continue
            print(file)
            df = jsonToDF(file)
            for index, row in df.iterrows():
                text = row['text']
                spans = row['spans']
                tokenList = sentences2tokens(text)
                tokens = [token.group() for token in tokenList]
                labels = list(repeat('O', len(tokenList)))
                for span in spans:
                    start = span["start"]
                    end = span["end"]
                    subtokens = []
                    for i, token in enumerate(tokenList):
                        if token.span()[1] > end:
                            break
                        if token.span()[0] >= start and token.span()[1] <= end:
                            subtokens.append((i, token))
                    for i, token in enumerate(subtokens):
                        index = token[0]
                        if i == 0:
                            labels[index] = 'B-Sentence'
                        elif i == len(subtokens)-1:
                            labels[index] = 'L-Sentence'
                        else:
                            labels[index] = 'I-Sentence'

                tokenlist.extend(tokens)
                labellist.extend(labels)

        all_data = []

        length = 0
        sen_count = 0
        sublist = list(zip(tokenlist, labellist))
        in_sentence = False
        subtokens = []
        sublabels = []
        for i, item in enumerate(sublist):
            length += 1
            if item[1] == 'B-Sentence':
                in_sentence = True
                sen_count += 1
            elif item[1] == 'L-Sentence':
                in_sentence = False
            if length > 300 and not in_sentence and sen_count >= 3:
                subtokens.append(item[0])
                sublabels.append(item[1])
                all_data.append({
                    'tokens': subtokens,
                    'labels': sublabels,
                    'length': len(subtokens)
                })
                subtokens = []
                sublabels = []

                length = 0
                sen_count = 0
                continue
            else:
                subtokens.append(item[0])
                sublabels.append(item[1])

            if i == len(sublist)-1:
                all_data.append({
                    'tokens': subtokens,
                    'labels': sublabels,
                    'length': len(subtokens)
                })

        self.data = pd.DataFrame.from_dict(all_data)
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        all_tokens = set()
        all_labels = set()
        sentences = []
        for index, row in self.data.iterrows():
            sentence = []
            for token in row['tokens']:
                all_tokens.add(token)
            for label in row['labels']:
                all_labels.add(label)
            for item in list(zip(row['tokens'], row['labels'])):
                sentence.append(item)
            sentences.append(sentence)
        all_tokens = list(all_tokens)
        all_labels = list(all_labels)
        json_tok = json.dumps(all_tokens)
        if not self.model_type == 'both':
            with open('utility/bilstm_corpus_'+"_".join(self.lang)+'_'+self.model_type+'/vocab.json', 'w') as file:
                file.write(json_tok)
            json_label = json.dumps(
                ['O', 'B-Sentence', 'L-Sentence', 'I-Sentence'])
            with open('utility/bilstm_corpus_'+"_".join(self.lang)+'_'+self.model_type+'/tags.json', 'w') as file:
                file.write(json_label)
        else:
            with open('utility/bilstm_corpus_'+"_".join(self.lang)+'/vocab.json', 'w') as file:
                file.write(json_tok)
            json_label = json.dumps(
                ['O', 'B-Sentence', 'L-Sentence', 'I-Sentence'])
            with open('utility/bilstm_corpus_'+"_".join(self.lang)+'/tags.json', 'w') as file:
                file.write(json_label)

        #self.data = self.data.drop(columns=['tokens', 'labels'])
        string = ""
        for index, row in self.data.iterrows():
            tokens = row['tokens']
            labels = row['labels']
            tok_string = json.dumps(tokens)
            lab_string = json.dumps(labels)
            line = tok_string+'\t'+lab_string+'\n'
            string = string+line

        with open(self.base_path+'/dataset.txt', 'w') as file:
            file.write(string)

    def finalize_training(self):
        args = [sys.executable, "-m",  "bi_lstm_crf", self.base_path, "--model_dir",
                self.model_dir, "--device", self.device, "--max_seq_len", self.max_seq_len, "--embedding_dim",
                self.embedding_dim, "--hidden_dim", self.hidden_dim, "--num_epoch", self.num_epoch,
                "--batch_size", self.batch_size, "--lr", self.lr, "--weight_decay", self.weight_decay,
                "--val_split", self.val_split, "--test_split", self.test_split]

        if self.recovery:
            args.extend(['--recovery'])
        if self.save_best_val_model:
            args.extend(['--save_best_val_model'])

        subprocess.run(args)

    def train(self):
        pass


class trainer_transformer(trainer):

    name = 'transformer'

    def __init__(self, langs: list, model_type: str = 'both') -> None:
        self.langs = langs
        self.model_type = model_type
        self.model_checkpoint = 'microsoft/Multilingual-MiniLM-L12-H384'#'distilbert-base-multilingual-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint,
            model_max_length=512,
            padding='max_length'
            )
        self.class_names = ["O", "B-Sentence", 'I-Sentence']
        self.features = Features({
            'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'labels': Sequence(feature=ClassLabel(num_classes=3, names=self.class_names), length=-1, id=None)
        })
        self.num_epochs = 5
        self.metric = evaluate.load("seqeval")

    def gather_text(self, files):
        token_docs = []
        label_docs = []
        for file in files:
            print(file)
            if "agb" in file:
                continue
            df = jsonToDF(file)
            # print(file)
            for y, row in df.iterrows():
                text = row['text']
                spans = row['spans']

                tokenList = sentences2tokens(text)
                span_collec = []
                current_collec = []
                total_length = 0
                for span in spans:
                    start = span['start']
                    end = span['end']

                    span_tokens = []

                    for token in tokenList:
                        if token.end() > end:
                            break
                        elif token.start() < start:
                            continue
                        elif token.start() >= start and token.end() <= end:
                            span_tokens.append(token.group())

                    span_length = len(self.tokenizer(
                        span_tokens, is_split_into_words=True).tokens())

                    if total_length + span_length > 300:
                        span_collec.append(current_collec)
                        current_collec = []
                        current_collec.append((start, end))
                        total_length = span_length
                    elif span == spans[-1]:
                        current_collec.append((start, end))
                        span_collec.append(current_collec)
                    else:
                        current_collec.append((start, end))
                        total_length += span_length

                tokenList = sentences2tokens(text)
                tokens = [token.group() for token in tokenList]
                labels = list(repeat('O', len(tokenList)))

                for spans in span_collec:
                    start = spans[0][0]
                    end = spans[-1][1]

                    for span in spans:

                        span_start = span[0]
                        span_end = span[1]
                        subtokens = []

                        for i, token in enumerate(tokenList):
                            if token.end() > span_end:
                                break
                            elif token.start() < span_start:
                                continue
                            elif token.start() >= span_start and token.end() <= span_end:
                                subtokens.append((i, token))
                        for i, token in enumerate(subtokens):
                            index = token[0]
                            if i == 0:
                                labels[index] = 'B-Sentence'
                            elif i == len(subtokens)-1:
                                labels[index] = 'I-Sentence'
                            else:
                                labels[index] = 'I-Sentence'

                    span_tokens = []
                    span_labels = []
                    for item in list(zip(tokenList, labels)):
                        if item[0].start() >= start and item[0].end() <= end:
                            span_tokens.append(item[0].group())
                            span_labels.append(item[1])

                    token_docs.append(span_tokens)
                    label_docs.append(span_labels)

        return Dataset.from_dict({
            'tokens': token_docs,
            'labels': label_docs
        }, features=self.features)

    def train_text(self):
        train_files = []
        test_files = []
        for lang in self.langs:
            if self.model_type == 'both':
                file_string = "*"
            elif self.model_type == 'judgements':
                file_string = "CD*"
            else:
                file_string = "*Code"
            train_files.extend(
                glob.glob('data/'+lang+'/gold/'+file_string+'_train.jsonl'))
            test_files.extend(
                glob.glob('data/'+lang+'/gold/'+file_string+'_test.jsonl'))

        train = self.gather_text(train_files)
        train = train.train_test_split(test_size=0.15, shuffle=False, seed=1234).shuffle()
        test = self.gather_text(test_files).shuffle()

        self.raw_datasets = DatasetDict({
            'train': train['train'],
            'test': test,
            'validation': train['test']
        })

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l]
                        for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p]
                for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                """if label % 2 == 1:
                    label += 1"""
                label = -100
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )
        all_labels = examples["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l]
                        for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p]
                for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def finalize_training(self):

        tokenized_datasets = self.raw_datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.raw_datasets['train'].column_names
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            max_length = 512,
            padding = 'max_length'
            )

        ner_feature = self.raw_datasets['train'].features['labels']
        self.label_names = ner_feature.feature.names

        id2label = {i: label for i, label in enumerate(self.label_names)}
        label2id = {v: k for k, v in id2label.items()}
        seed = np.random.randint(10000)
        set_seed(seed)

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            id2label=id2label,
            label2id=label2id
        )

        assert model.config.num_labels == 3

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=8
        )

        eval_dataloader = DataLoader(
            tokenized_datasets["validation"],
            collate_fn=data_collator,
            batch_size=8
        )

        optimizer = AdamW(model.parameters(), lr=2e-5)
 
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = self.num_epochs
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        if self.model_type == 'both':
            model_name = 'minilm-SBD-' + \
                "-".join(self.langs)+'-judgements-laws'
        else:
            model_name = 'minilm-SBD-' + \
                "-".join(self.langs)+'-'+self.model_type

        repo_name = get_full_repo_name(model_name)
        output_dir = 'models/'+model_name
        repo = Repository(output_dir, clone_from=repo_name)

        progress_bar = tqdm(range(num_training_steps))
        losses = []
        val_losses = []
        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(outputs.loss.tolist())
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                val_losses.append(outputs.loss.tolist())
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)

                true_predictions, true_labels = self.postprocess(
                    predictions_gathered, labels_gathered)
                self.metric.add_batch(
                    predictions=true_predictions, references=true_labels)

            results = self.metric.compute()
            print(
                f"epoch {epoch}:",
                {
                    key: round(results[f"overall_{key}"], 3)
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, save_function=accelerator.save, )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                """repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False
                )"""
        if accelerator.is_main_process:
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False)
        df = pd.DataFrame.from_dict({
            'loss': [losses],
            'val_loss': [val_losses]
        })
        # df.to_csv(output_dir+'/losses.csv')

    def train(self):
        pass


"""class trainer_(trainer):

    name = 'trainer_nltk'

    def __init__(self, lang) -> None:
        self.lang = lang

    def train_text(self):
        pass

    def finalize_training(self):
        pass

    def train(self):
        pass"""
