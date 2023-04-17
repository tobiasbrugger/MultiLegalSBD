# MultiLegalSBD

Code and models for my paper "MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset" found at: -

The transformers models can also be found at: found here: https://huggingface.co/models?search=rcds/distilbert-sbd

The data can also be found at: https://huggingface.co/datasets/rcds/MultiLegalSBD

The code was used to train and evaluate CRF, BiLSTM-CRF and transformer models on legal data in French, Spanish, Italian, German and English.

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
