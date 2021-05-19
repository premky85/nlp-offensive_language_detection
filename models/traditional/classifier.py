import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from imblearn.over_sampling import SMOTE

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

if __name__ == "__main__":
    # variables
    CLASSIFIER_TYPE = 'LOGISTIC_REGRESSION'
    OVERSAMPLE = False

    # read the data and features
    tweet_data = pd.read_csv('../../data/preprocessed_data/twitter_preprocessed.csv')
    tfidf_features = pd.read_csv('data/tfidf_features.csv')
    bigram_features = pd.read_csv('data/bigram_features.csv')
    sentiment_scores = pd.read_csv('data/sentiment_scores.csv')

    features = tweet_data[['index', 'is_offensive']]
    # merge other features
    features = features.merge(sentiment_scores, on='index')
    features = features.merge(bigram_features, on='index')
    features = features.merge(tfidf_features, on='index')

    # checking if data is imbalanced (uncomment to see chart)
    # sns.countplot(x='is_offensive', data=features, palette='hls')
    # plt.show()

    # split into class labels and features
    y = features.iloc[:, 1]
    X = features.iloc[:, 2:]

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if OVERSAMPLE:
        # we do oversampling only on training data
        oversampler = SMOTE(random_state=0)
        columns = X_train.columns
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(data=X_train, columns=columns)
        y_train = pd.DataFrame(data=y_train, columns=['is_hate'])

    # create model
    if CLASSIFIER_TYPE == 'LOGISTIC_REGRESSION':
        model = LogisticRegression()
    elif CLASSIFIER_TYPE == 'SVM':
        model = LinearSVC(max_iter=1000000)
    elif CLASSIFIER_TYPE == 'RANDOM_FOREST':
        model = RandomForestClassifier()
    elif CLASSIFIER_TYPE == 'DUMMY':
        model = DummyClassifier(strategy='most_frequent')

    # fit and predict, then generate confusion matrix
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    print(f"CLASSIFIER: {CLASSIFIER_TYPE}")
    print("CONFUSION MATRIX:")
    print(cm)
    print(f"ACCURACY: {precision_score(y_test, predictions)}")
    print(classification_report(y_test, predictions))
