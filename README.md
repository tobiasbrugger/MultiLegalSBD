# MultiLegalSBD

Code and models for my paper "MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset" found at: 
The transformers models can also be found at: https://huggingface.co/tbrugger
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
