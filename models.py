import glob
import os
import pickle
import subprocess
import sys
from itertools import repeat
from pathlib import Path
from signal import SIGTERM
from subprocess import PIPE, Popen

import nltk.data
import pycrfsuite
import pysbd
import spacy
import stanza
from bi_lstm_crf.app import WordsTagger
from huggingface_hub import hf_hub_download, snapshot_download
from psutil import process_iter
from spacy import Language
from spacy.cli.train import train
from stanza.server import CoreNLPClient
from transformers import AutoTokenizer, pipeline

from crf_features import CRF_Features
from jsonToDF import jsonToDF
from luima_sbd.sbd_utils import text2sentences
from tokenizer import sentences2tokens, sentences2tokensWithWS


class module(object):
    """
    Base class for models
    """

    def __init__(self, lang: str) -> None:
        """
        :param str lang: The language the model should be initialized for
        """
        raise NotImplementedError

    def sentenize(self, text) -> list[tuple[str, int]]:
        """
        Returns list of tokenized text with its label
        :param str text: The text to be predicted
        :return: List of tuple of token, label
        :rtype: List[tuple(str,int)]
        """
        raise NotImplementedError


def spans2labels(text: str, spans: list) -> list:
    """
    Converts a list of spans to a list of tokens to be evaluated
    :param str text: The text that was predicted
    :param list[tuple(int,int)] spans: the predicted spans
    :return: list of tokens with its associated token
    :rtype: list[tuple(str, int)]
    """
    tokens = sentences2tokens(text)
    labels = list(repeat(0, len(tokens)))
    for span in spans:
        start = span[0]
        end = span[1]
        subtokens = [(i, subtoken) for i, subtoken in enumerate(
            tokens) if subtoken.start() >= start and subtoken.end() <= end]
        for i, subtoken in subtokens:
            if subtoken == subtokens[0][1]:
                labels[i] = 1
            elif subtoken == subtokens[-1][1]:
                labels[i] = 1
            else:
                labels[i] = 0

    tokens = [token.group() for token in tokens]
    return list(zip(tokens, labels))


class mod_nltk(module):

    name = "nltk"
    lang = ''

    def __init__(self, lang, model_type) -> None:
        self.lang = lang[0]
        match self.lang:
            case 'fr':
                self.model = nltk.data.load('models/french.pickle')
            case 'it':
                self.model = nltk.data.load('models/italian.pickle')
            case 'es':
                self.model = nltk.data.load('models/spanish.pickle')
            case 'en':
                self.model = nltk.data.load('models/english.pickle')
            case 'de':
                self.model = nltk.data.load('models/german.pickle')
            case 'br':
                self.model = nltk.data.load('models/english.pickle')
            case _:
                self.model = nltk.data.load('models/french.pickle')

    def sentenize(self, text):
        spans = self.model.span_tokenize(text)
        return spans2labels(text, spans)


class mod_nltk_train(module):

    name = "nltk_train"
    lang = ''
    trainer = None
    model = None
    model_path = ''

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang[0]
        self.model_path = 'models/nltk_model_'+self.lang+'_trained.mdl'
        if os.path.exists(self.model_path):
            self.load_model()

    def load_model(self):
        assert os.path.isfile(self.model_path)
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def sentenize(self, text):
        spans = self.model.span_tokenize(text)
        return spans2labels(text, spans)

    def sentenize_spans(self, text):
        spans = self.model.span_tokenize(text)
        return spans


class mod_luima_sbd(module):

    name = "luima_sbd"
    lang = ''

    def __init__(self, lang, model_type) -> None:
        self.lang = lang[0]

    def sentenize(self, text):
        spans = text2sentences(text, offsets=True)
        return spans2labels(text, spans)

    def sentenize_spans(self, text):
        spans = text2sentences(text, offsets=True)
        return spans


class mod_coreNLP(module):

    name = "coreNLP"
    port = 9000
    props = {
        'tokenize.whitespace': True,
        'tokenize.keepeol': True
    }
    client = None
    lang = ''

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang[0]
        match lang:
            case 'fr':
                path = Path(
                    'utility\corenlp\stanford-corenlp-4.5.1-models-french.jar')
                if not path.exists():
                    self.init_setup('french')
            case 'it':
                path = Path(
                    'utility\corenlp\stanford-corenlp-4.5.1-models-italian.jar')
                if not path.exists():
                    self.init_setup('italian')
            case 'es':
                path = Path(
                    'utility\corenlp\stanford-corenlp-4.5.1-models-spanish.jar')
                if not path.exists():
                    self.init_setup('spanish')
            case 'en':
                path = Path(
                    'utility\corenlp\stanford-corenlp-4.5.1-models-english-kbp.jar')
                if not path.exists():
                    self.init_setup('english-kbp')
            case 'de':
                path = Path(
                    'utility\corenlp\stanford-corenlp-4.5.1-models-german.jar')
                if not path.exists():
                    self.init_setup('german')

        if not self.client:
            self.client = CoreNLPClient(
                annotators=['tokenize', 'ssplit'],
                properties=self.lang,
                timeout=60000,
                memory='6G',
                output_format='json',
                max_char_length=500000,
                be_quiet=True,
                # start_server='TRY_START',
                endpoint="http://localhost:"+str(self.port)
            )

    def init_setup(self, language):

        # stanza.install_corenlp(dir='utility/corenlp')
        stanza.download_corenlp_models(
            model=language, version='4.5.1', dir='utility/corenlp')

    def sentenize(self, text):

        sentences = self.client.annotate(text)
        spans = []
        for sen in sentences["sentences"]:
            start = sen["tokens"][0]['characterOffsetBegin']
            end = sen["tokens"][-1]['characterOffsetEnd']
            if sen == sentences['sentences'][-1]:
                end += 1
            spans.append((start, end))
        return spans2labels(text, spans)


class mod_stanza(module):
    name = "stanza"
    nlp = None
    lang = ''

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang[0]
        if self.lang == 'br':
            self.lang = 'pt'
        self.nlp = stanza.Pipeline(lang=self.lang, processors='tokenize')

    def sentenize(self, text):
        doc = self.nlp(text)
        spans = [(sent.tokens[0].start_char, sent.tokens[-1].end_char)
                 for sent in doc.sentences]
        return spans2labels(text, spans)


class mod_spacy(module):

    name = "spacy"
    model = None
    lang = ''
    # fr_core_news_sm

    def __init__(self, lang, model_type='both') -> None:
        self.lang = lang[0]

        match self.lang:
            case 'fr':
                self.model = spacy.load('fr_dep_news_trf')  # fr_dep_news_trf
            case 'it':
                self.model = spacy.load('it_core_news_lg')
            case 'es':
                self.model = spacy.load('es_dep_news_trf')
            case 'de':
                self.model = spacy.load('de_dep_news_trf')
            case 'en':
                self.model = spacy.load('en_core_web_trf')
            case 'br':
                self.model = spacy.load('pt_core_news_lg')
            case _:
                self.model = spacy.load('fr_core_news_sm')

        self.model.add_pipe("custom_seg", before='parser')

    @Language.component('custom_seg')
    def custom_seg(doc):
        for token in doc:
            if len(doc) >= token.i:
                continue
            if token.text == '\n' and (token.is_sent_start == True or token.is_sent_start == None):
                token.is_sent_start = False
                doc[token.i+1].is_sent_start = True

        return doc

    def sentenize(self, text):
        doc = self.model(text)

        spans = []
        for sentence in doc.sents:
            temp = sentence.text.rstrip('\n')
            length = len(sentence.text) - len(temp)
            spans.append((sentence.start_char, sentence.end_char-length))
        return spans2labels(text, spans)

    def sentenize_spans(self, text):
        doc = self.model(text)
        spans = []
        for sentence in doc.sents:
            spans.append((sentence.start_char, sentence.end_char))
        return spans


class mod_transformer(module):

    name = "transformer"
    pipe = None
    lang = ''

    def __init__(self, lang, model_type='both') -> None:
        self.model_type = model_type
        self.lang = "-".join(lang)

        if self.model_type == 'both':
            self.model_path = 'tbrugger/distilbert-SBD-'+self.lang+'-judgements-laws'
        else:
            self.model_path = 'tbrugger/distilbert-SBD-' + \
                self.lang+'-'+self.model_type

        self.pipe = pipeline(
            'token-classification',
            model=self.model_path,
            aggregation_strategy="simple",  # none, simple, first, average, max
            use_auth_token=True,
            device = 0
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def preds2sentences(self, matches, preds):
        indices = []
        in_annotation = False
        start, end = (0, 0)
        for label, match in zip(preds, matches):
            if label != 'O':
                if in_annotation:
                    end = match.end()
                else:
                    in_annotation = True
                    start = match.start()
                    end = match.end()
            else:
                if in_annotation:
                    in_annotation = False
                    indices.append((start, end))
        if in_annotation:
            indices.append((start, end))
        return indices

    def sentenize(self, text, spans):

        tokenList = sentences2tokens(text)
        tokens = [token.group() for token in tokenList]
        labels = list(repeat(0, len(tokenList)))
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
        

        for i, spans in enumerate(span_collec):
            start = spans[0][0]
            end = spans[-1][1]
            if spans != span_collec[-1]:
                end = span_collec[i+1][0][0] #correct span end to next start to not lose characters outside of gold spans

            sentences = self.pipe(text[start:end])

            for sentence in sentences:
                sen_start = sentence['start']+start
                sen_end = sentence['end']+start
                subtokens = []
                for i, token in enumerate(tokenList):
                    if token.end() > sen_end:
                        break
                    elif token.start() < sen_start:
                        continue
                    elif token.start() >= sen_start and token.end() <= sen_end:
                        subtokens.append((i, token))
                for i, token in enumerate(subtokens):
                    index = token[0]
                    if i == 0:
                        labels[index] = 1
                    elif i == len(subtokens)-1:
                        labels[index] = 1
                    else:
                        labels[index] = 0


        """for y,item in enumerate(texts):
            start = item['spans'][0][0]
            end = item['spans'][-1][1]
            sentences = self.pipe(item['text'])

            for sentence in sentences:
                sen_start = sentence['start']+start
                sen_end = sentence['end']+start
                subtokens = []
                for i, token in enumerate(tokenList):
                    if token.end() > sen_end:
                        break
                    elif token.start() < sen_start:
                        continue
                    elif token.start() >= sen_start and token.end() <= sen_end:
                        subtokens.append((i, token))
                for i, token in enumerate(subtokens):
                    index = token[0]
                    if i == 0:
                        labels[index] = 1
                    elif i == len(subtokens)-1:
                        labels[index] = 1
                    else:
                        labels[index] = 0"""

        return list(zip(tokens, labels))

    def sentenize_spans(self, text, spans):
        tokenList = sentences2tokensWithWS(text)
        tokens = [token.group() for token in tokenList]
        labels = list(repeat('O', len(tokenList)))
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
        texts = []
        for i, spans in enumerate(span_collec):
            start = spans[0][0]
            end = spans[-1][1]
            if spans != span_collec[-1]:
                if end+1 != span_collec[i+1][0][0]:
                    end = end+1

            spans_text = text[start:end]
            texts.append({
                'text': spans_text,
                'spans': spans
            })

        for item in texts:
            start = item['spans'][0][0]
            end = item['spans'][-1][1]
            sentences = self.pipe(item['text'])
            for sentence in sentences:
                sen_start = sentence['start']+start
                sen_end = sentence['end']+start
                subtokens = []
                for i, token in enumerate(tokenList):
                    if token.end() > sen_end:
                        break
                    elif token.start() < sen_start:
                        continue
                    elif token.start() >= sen_start and token.end() <= sen_end:
                        subtokens.append((i, token))
                for i, token in enumerate(subtokens):
                    index = token[0]
                    if i == 0:
                        labels[index] = "B-Sentence"
                    elif i == len(subtokens)-1:
                        labels[index] = "L-Sentence"
                    else:
                        labels[index] = "I-Sentence"
        assert len(tokenList) == len(labels)
        return self.preds2sentences(tokenList, labels)

    def sentenize_test(self, text):
        sentences = self.pipe(text)
        return [(sen['start'], sen['end']) for sen in sentences]


class mod_crf(module):

    name = "crf"
    tagger = None
    features = None
    lang_string = ''
    model_path = ''

    def __init__(self, lang, model_type='both') -> None:
        self.lang_string = "_".join(lang)
        self.model_type = model_type
        if self.model_type == 'laws':
            self.model_path = 'models/crf_'+self.lang_string+'_' + self.model_type+'.crfsuite'
        elif self.model_type == 'judgements':
            self.model_path = 'models/crf_'+self.lang_string+'_' + self.model_type+'.crfsuite'
        else:
            self.model_path = 'models/crf_'+self.lang_string+'.crfsuite'
        self.features = CRF_Features()
        self.load_model()

    def load_model(self):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model_path)

    def preds2sentences(self, matches, preds):
        indices = []
        in_annotation = False
        start, end = (0, 0)
        for label, match in zip(preds, matches):
            if label != 'O':
                if in_annotation:
                    end = match.end()
                else:
                    in_annotation = True
                    start = match.start()
                    end = match.end()
            else:
                if in_annotation:
                    in_annotation = False
                    indices.append((start, end))
        if in_annotation:
            indices.append((start, end))
        return indices

    def sentenize(self, text):
        tokenList = sentences2tokensWithWS(text)
        tokens = [token.group() for token in tokenList]
        feat = self.features.generate_features(tokens)
        labels = self.tagger.tag(feat)
        return spans2labels(text, self.preds2sentences(tokenList, labels))

    def sentenize_spans(self, text):
        tokenList = sentences2tokensWithWS(text)
        tokens = [token.group() for token in tokenList]
        feat = self.features.generate_features(tokens)
        labels = self.tagger.tag(feat)
        return self.preds2sentences(tokenList, labels)

    def sentenize_original(self, text):
        tokens = [token.group() for token in sentences2tokensWithWS(text)]
        feat = self.features.generate_features(tokens)
        labels = self.tagger.tag(feat)
        result = []
        assert len(tokens) == len(labels)
        for item in list(zip(tokens, labels)):
            if " " in item[0]:
                continue
            if item[1] == 'B-Sentence':
                result.append((item[0], 1))
            elif item[1] == 'L-Sentence':
                result.append((item[0], 1))
            else:
                result.append((item[0], 0))
        return result


class mod_bilstm_crf(module):
    name = "bilstm_crf"
    tagger = None

    def __init__(self, lang, model_type='both') -> None:
        self.lang = "_".join(lang)
        self.model_type = model_type
        if self.model_type == 'judgements':
            basepath = 'models/bilstm_crf_'+self.lang+'_'+self.model_type
        elif self.model_type == 'laws':
            basepath = 'models/bilstm_crf_'+self.lang+'_'+self.model_type
        else:
            basepath = 'models/bilstm_crf_'+self.lang
        self.tagger = WordsTagger(
            basepath,
            device='cuda:0'#'cuda:0'
        )

    def sentenize(self, text, spans):
        tokens = [token.group() for token in sentences2tokens(text)]

        tags, seq = self.tagger([tokens])
        labels = tags[0]
        binary_labels = []
        for label in labels:
            if label == 'B-Sentence':
                binary_labels.append(1)
            elif label == 'L-Sentence':
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        assert len(tokens) == len(binary_labels)
        return list(zip(tokens, binary_labels))

    def sentenize_inference(self, text):
        pass