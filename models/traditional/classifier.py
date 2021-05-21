import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

if __name__ == "__main__":
    # variables
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
    y_train = features.iloc[:, 1]
    X_train = features.iloc[:, 2:]

    # split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # uncomment if you dont use cross validation

    if OVERSAMPLE:
        # we do oversampling only on training data
        oversampler = SMOTE(random_state=0)
        columns = X_train.columns
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(data=X_train, columns=columns)
        y_train = pd.DataFrame(data=y_train, columns=['is_offensive'])

    # create models
    print("Creating models...")
    logistic_regression = LogisticRegression()
    support_vector = LinearSVC(max_iter=100000)
    random_forest = RandomForestClassifier()
    dummy = DummyClassifier(strategy='most_frequent')
    extreme_gradient_boost = XGBClassifier(learning_rate=.025, max_features=100)

    # cross validation for all models
    FOLD_COUNT = 5
    SCORING = ['precision', 'recall', 'f1', 'f1_macro']
    print("Performing cross validation...")
    print("validating logistic regression...")
    logistic_regression_scores = cross_validate(logistic_regression, X_train, y_train, cv=FOLD_COUNT, scoring=SCORING)
    print("validating support vector...")
    support_vector_scores = cross_validate(support_vector, X_train, y_train, cv=FOLD_COUNT, scoring=SCORING)
    print("validating eXtreme gradient boost...")
    extreme_gradient_boost_scores = cross_validate(extreme_gradient_boost, X_train, y_train, cv=FOLD_COUNT, scoring=SCORING)
    print("validating random forest...")
    random_forest_scores = cross_validate(random_forest, X_train, y_train, cv=FOLD_COUNT, scoring=SCORING)
    print("validating dummy model...")
    dummy_scores = cross_validate(dummy, X_train, y_train, cv=FOLD_COUNT, scoring=SCORING)

    print("SCORES:")
    lr_mean_scores = [logistic_regression_scores['test_precision'].mean(), logistic_regression_scores['test_recall'].mean(),
                    logistic_regression_scores['test_f1'].mean(), logistic_regression_scores['test_f1_macro'].mean()]
    sv_mean_scores = [support_vector_scores['test_precision'].mean(), support_vector_scores['test_recall'].mean(),
                    support_vector_scores['test_f1'].mean(), support_vector_scores['test_f1_macro'].mean()]
    egb_mean_scores = [extreme_gradient_boost_scores['test_precision'].mean(), extreme_gradient_boost_scores['test_recall'].mean(),
                    extreme_gradient_boost_scores['test_f1'].mean(), extreme_gradient_boost_scores['test_f1_macro'].mean()]
    rf_mean_scores = [random_forest_scores['test_precision'].mean(), random_forest_scores['test_recall'].mean(),
                    random_forest_scores['test_f1'].mean(), random_forest_scores['test_f1_macro'].mean()]
    d_mean_scores = [dummy_scores['test_precision'].mean(), dummy_scores['test_recall'].mean(),
                    dummy_scores['test_f1'].mean(), dummy_scores['test_f1_macro'].mean()]
    
    print(f"LOGISTIC REGRESSION: PRECISION: {lr_mean_scores[0]}, RECALL: {lr_mean_scores[1]}, F1: {lr_mean_scores[2]}, F1_MACRO: {lr_mean_scores[3]}")
    print(f"SUPPORT VECTOR: PRECISION: {sv_mean_scores[0]}, RECALL: {sv_mean_scores[1]}, F1: {sv_mean_scores[2]}, F1_MACRO: {sv_mean_scores[3]}")
    print(f"EXTREME GRADIENT BOOST: PRECISION: {egb_mean_scores[0]}, RECALL: {egb_mean_scores[1]}, F1: {egb_mean_scores[2]}, F1_MACRO: {egb_mean_scores[3]}")
    print(f"RANDOM FOREST: PRECISION: {rf_mean_scores[0]}, RECALL: {rf_mean_scores[1]}, F1: {rf_mean_scores[2]}, F1_MACRO: {rf_mean_scores[3]}")
    print(f"DUMMY MODEL: PRECISION: {d_mean_scores[0]}, RECALL: {d_mean_scores[1]}, F1: {d_mean_scores[2]}, F1_MACRO: {d_mean_scores[3]}")

    #if INCLUDE_BOOST_MODEL:
    #    extreme_gradient_boost = XGBClassifier(learning_rate=.025, max_features=100)
    #    print("boosting method score:", cross_val_score(extreme_gradient_boost, X_train, y_train, cv=5, scoring="f1_micro").mean())
