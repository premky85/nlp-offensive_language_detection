
# Getting started
This repository contains different approaches to English-Slovenian Offensive language detection. You can find different models in `/models` folder.

## Requirements

Installing Slovene stemmer:
```sh
pip install git+https://repo.ijs.si/DIS-AGENTS/snowball-stemmer
```

## Data sources and preprocessing

1. Download csv files from links below and save them in **data/raw** folder:
    - [Gab & Reddit datasets](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/tree/master/data)
    - [English twitter dataset](https://data.world/thomasrdavidson/hate-speech-and-offensive-language)
    - [Slovenian twitter dataset (download third file - IMSyPP_SI_anotacije_evaluation-clarin.csv)](https://www.clarin.si/repository/xmlui/handle/11356/1398)
    - [White supremacist forum (download **all_files** folder and **annotations_metadata.csv**)](https://github.com/Vicomtech/hate-speech-dataset)
2. Contact BLAŽ RUPNIK/Maj Šavli/Leon Premk on Slack for Twitter api credentials
3. Paste the credentials in the script **data/preprocessing/data_preprocessing.py** (lines 139-142)
3. Run the script

## Usage

### Traditional approaches

For regenerating features use (make sure preprocessed dataset is in correct folder):
- `preprocess_ngram.py` for generating bigrams
- `preprocess_sentiment.py` for sentiment scores
- `preprocess_tfidf.py` for tf-idf weighted scores

For classification you can use:
- `multiclass_classifier.py` for multi class prediction evaluation
- `classifier.py` for binary (is_offensive) prediction evaluation

For ROC visualization of binary predictor you can use `visualize_binary.py`.

### Deep neural networks 
Example of DNN model usage can be foud in `predict.ipynb`. Trained models can be obtained [HERE](https://drive.google.com/drive/folders/1xRx20enFwD3mVe8LvX2c8v5M37jvo0X5?usp=sharing). Password can be obtained by contacting me on email: <lp8783@student.uni-lj.si>.

There are 3 different architectures:
- mBERT
- XLMr
- CroSloEngualBERT

Each trained on three different datasets:
- Slovenian + English
- Slovenian
- English
