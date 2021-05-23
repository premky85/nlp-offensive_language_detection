# Traditional approaches
##### This directory consists of preprocess scripts, classifiers and data folder where generated features are stored 

For regenerating features use (make sure preprocessed dataset is in correct folder):
- `preprocess_ngram.py` for generating bigrams
- `preprocess_sentiment.py` for sentiment scores
- `preprocess_tfidf.py` for tf-idf weighted scores

For classification you can use:
- `multiclass_classifier.py` for multi class prediction evaluation
- `classifier.py` for binary (is_offensive) prediction evaluation

For ROC visualization of binary predictor you can use `visualize_binary.py`.

We suggest using Python version 3.8. Additionaly you need some libraries that you can install with pip:
```sh
pip install pandas sklearn xgboost matplotlib sns imblearn nltk 
```
