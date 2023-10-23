# MultiLegalSBD

Code and models for my paper "MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset" found at: -

The transformers models can also be found at: https://huggingface.co/models?search=rcds/distilbert-sbd

The datasets can also be found at: https://huggingface.co/datasets/rcds/MultiLegalSBD

The code was used to train and evaluate CRF, BiLSTM-CRF and transformer models on legal data in French, Spanish, Italian, German, English and Portuguese.

## Disclaimer

## Installation

Install dependencies using:
```py
pip install -r requirements.txt
```

For Pytorch (see: https://pytorch.org/):
```py
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

## Models
### Examples
for training see https://github.com/tobiasbrugger/MultiLegalSBD/blob/master/trainer.py

for model usage see https://github.com/tobiasbrugger/MultiLegalSBD/blob/master/models.py

### Transformer Models 
(found here: https://huggingface.co/models?search=rcds/distilbert-sbd)

```py
from transformers import pipeline

pipe = pipeline(
  'token-classification',
  model= '<Organization/ModelName>', #HuggingFace URL e.g. rcds/distilbert-SBD-fr-judgements-laws
  aggregation_strategy="simple",  # none, simple, first, average, max
  device = 0 #use GPU
)

text = 'This is a sentence.'
sentences = pipe(text)
```

#### BUG
If you get the following error:

RuntimeError: Failed to import transformers.models.distilbert.modeling_distilbert because of the following error (look up to see its traceback):
module 'signal' has no attribute 'SIGKILL'

Changing 'SIGKILL' to 'SIGTERM' in the file 'file_based_local_timer.py' (where the bug occurs) fixed it for me.

### BiLSTM-CRF Models
(found here: https://github.com/tobiasbrugger/MultiLegalSBD/tree/master/models)

requires installed bi_lstm_crf library (https://github.com/jidasheng/bi-lstm-crf)

```py
from bi_lstm_crf.app import WordsTagger
from tokenizer import sentences2tokens

tagger = WordsTagger(
  '<PathToModel>', # e.g. models/bilstm_crf_de
  device = 'cuda:0' # use GPU, cpu:0 alternatively
  )

text = 'This is a sentence.'
tokens = [token.group() for token in sentences2tokens(text)] # bi_lstm_crf requires pre-tokenized input text
labels, sequence = tagger([tokens])
```

### CRF Models
(found here: https://github.com/tobiasbrugger/MultiLegalSBD/tree/master/models)

requires installed pycrfsuite (https://pypi.org/project/python-crfsuite/)
```py
import pycrfsuite
from crf_features import CRF_Features
from tokenizer import sentences2tokens

features = CRF_Features() # CRF features generator
tagger = pycrfsuite.Tagger()
tagger.open('<PathToModel>') # e.g. models/crf_de.crfsuite

text = 'This is a sentence.'

tokens = [token.group() for token in sentences2tokens(text)] 
feat = features.generate_features(tokens) # pycrfsuite takes features as input rather than tokenized text
labels = tagger.tag(feat)
```

# Citation
´´´
@inproceedings{10.1145/3594536.3595132,
author = {Brugger, Tobias and St\"{u}rmer, Matthias and Niklaus, Joel},
title = {MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset},
year = {2023},
isbn = {9798400701979},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3594536.3595132},
doi = {10.1145/3594536.3595132},
abstract = {Sentence Boundary Detection (SBD) is one of the foundational building blocks of Natural Language Processing (NLP), with incorrectly split sentences heavily influencing the output quality of downstream tasks. It is a challenging task for algorithms, especially in the legal domain, considering the complex and different sentence structures used. In this work, we curated a diverse multilingual legal dataset consisting of over 130'000 annotated sentences in 6 languages. Our experimental results indicate that the performance of existing SBD models is subpar on multilingual legal data. We trained and tested monolingual and multilingual models based on CRF, BiLSTM-CRF, and transformers, demonstrating state-of-the-art performance. We also show that our multilingual models outperform all baselines in the zero-shot setting on a Portuguese test set. To encourage further research and development by the community, we have made our dataset, models, and code publicly available.},
booktitle = {Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law},
pages = {42–51},
numpages = {10},
keywords = {Natural Language Processing, Sentence Boundary Detection, Text Annotation, Legal Document Analysis, Multilingual},
location = {Braga, Portugal},
series = {ICAIL '23}
}
´´´
