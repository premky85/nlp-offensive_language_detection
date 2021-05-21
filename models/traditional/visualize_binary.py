import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

if __name__ == "__main__":
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

    # split into class labels and features
    y = features.iloc[:, 1]
    X = features.iloc[:, 2:]

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # create models
    print("Creating models...")
    logistic_regression = LogisticRegression(max_iter=1000)
    support_vector = LinearSVC(max_iter=100000)
    clf_sv = CalibratedClassifierCV(support_vector)
    random_forest = RandomForestClassifier()
    dummy = DummyClassifier(strategy='most_frequent')
    extreme_gradient_boost = XGBClassifier(learning_rate=.025, max_features=100)

    logistic_regression.fit(X_train, y_train)
    clf_sv.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    dummy.fit(X_train, y_train)
    extreme_gradient_boost.fit(X_train, y_train)

    # predict probabilities
    print("Prediction probabilities...")
    logistic_probabilities = logistic_regression.predict_proba(X_test)
    support_probabilities = clf_sv.predict_proba(X_test)
    forest_probabilities = random_forest.predict_proba(X_test)
    dummy_probabilities = dummy.predict_proba(X_test)
    egb_probabilities = extreme_gradient_boost.predict_proba(X_test) 


    # extract only for the positive outcome
    logistic_probabilities = logistic_probabilities[:, 1]
    support_probabilities = support_probabilities[:, 1]
    forest_probabilities = forest_probabilities[:, 1]
    dummy_probabilities = dummy_probabilities[:, 1]
    egb_probabilities = egb_probabilities[:, 1]

    # false positive and true positive
    fpr_logistic, tpr_logistic, _ = roc_curve(y_test, logistic_probabilities)
    fpr_support, tpr_support, _ = roc_curve(y_test, support_probabilities)
    fpr_forest, tpr_forest, _ = roc_curve(y_test, forest_probabilities)
    fpr_dummy, tpr_dummy, _ = roc_curve(y_test, dummy_probabilities)
    fpr_egb, tpr_egb, _ = roc_curve(y_test, egb_probabilities)

    # plot
    print("Plotting...")
    plt.plot(fpr_logistic, tpr_logistic, label='Logistic Regression')
    plt.plot(fpr_support, tpr_support, label="Support Vector Machine")
    plt.plot(fpr_forest, tpr_forest, label="Random Forest")
    plt.plot(fpr_dummy, tpr_dummy, label="Dummy")
    plt.plot(fpr_egb, tpr_egb, label='eXtreme Gradient Boosting')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
