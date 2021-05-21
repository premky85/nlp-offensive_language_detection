import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import StandardScaler
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

    features = tweet_data[['index', 'class']]
    # merge other features
    features = features.merge(sentiment_scores, on='index')
    features = features.merge(bigram_features, on='index')
    features = features.merge(tfidf_features, on='index')

    # checking if data is imbalanced (uncomment below to see chart)
    # sns.countplot(x='class', data=features, palette='hls')
    # plt.show()

    # split into class labels and features
    y = features.iloc[:, 1]
    X = features.iloc[:, 2:]

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # create models
    print("Creating models...")
    logistic_regression = LogisticRegression(max_iter=1000)
    support_vector = LinearSVC(max_iter=100000)
    random_forest = RandomForestClassifier()
    dummy = DummyClassifier(strategy='most_frequent')
    extreme_gradient_boost = XGBClassifier(learning_rate=.025, max_features=100)

    # feature scaling
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # fit
    print("Fitting models...")
    logistic_regression.fit(X_train, y_train)
    support_vector.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    dummy.fit(X_train, y_train)
    extreme_gradient_boost.fit(X_train, y_train)
    
    # predict
    print("Creating predictions...")
    print("logistic regression...")
    y_pred_lr = logistic_regression.predict(X_test)
    print("support vector...")
    y_pred_sv = support_vector.predict(X_test)
    print("random forest...")
    y_pred_rf = random_forest.predict(X_test)
    print("dummy...")
    y_pred_d = dummy.predict(X_test)
    print("extreme gradient boosting...")
    y_pred_egb = extreme_gradient_boost.predict(X_test)
    
    # print results
    print(f"LOGISTIC REGRESSION: {precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')}")
    print(f"SUPPORT VECTOR: {precision_recall_fscore_support(y_test, y_pred_sv, average='weighted')}")
    print(f"RANDOM FOREST: {precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')}")
    print(f"DUMMY: {precision_recall_fscore_support(y_test, y_pred_d, average='weighted')}")
    print(f"eXtreme GRADIENT BOOST: {precision_recall_fscore_support(y_test, y_pred_egb, average='weighted')}")

    # f1 macro
    print(f"LOGISTIC REGRESSION F1 MACRO: {f1_score(y_test, y_pred_lr, average='macro')}")
    print(f"SUPPORT VECTOR F1 MACRO: {f1_score(y_test, y_pred_sv, average='macro')}")
    print(f"RANDOM FOREST F1 MACRO: {f1_score(y_test, y_pred_rf, average='macro')}")
    print(f"DUMMY F1 MACRO: {f1_score(y_test, y_pred_d, average='macro')}")
    print(f"eXtreme GRADIENT BOOST F1 MACRO: {f1_score(y_test, y_pred_egb, average='macro')}")
