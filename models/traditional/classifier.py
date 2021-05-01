import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score

if __name__ == "__main__":
    # read the data and features
    tweet_data = pd.read_csv('../../data/twitter/twitter_preprocessed.csv')
    tfidf_features = pd.read_csv('data/tfidf_features.csv')
    bigram_features = pd.read_csv('data/bigram_features.csv')
    sentiment_scores = pd.read_csv('data/sentiment_scores.csv')

    features = tweet_data[['index', 'is_hate']]
    # merge other features
    features = features.merge(bigram_features, on='index')
    features = features.merge(tfidf_features, on='index')
    features = features.merge(sentiment_scores, on='index')

    # split into class labels and features
    y = features.iloc[:, 1]
    X = features.iloc[:, 2:]

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create model
    model = LogisticRegression()

    # fit and predict, then generate confusion matrix
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print(precision_score(y_test, predictions))
