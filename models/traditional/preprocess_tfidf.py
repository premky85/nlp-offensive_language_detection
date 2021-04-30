import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv('../../data/twitter/twitter_preprocessed.csv')

    snow_stemmer = SnowballStemmer("english") # language is required
    data['stemmed'] = data.tweet.map(lambda tweet: ' '.join([snow_stemmer.stem(word) for word in str(tweet).split(' ')]))

    count_vectorizer = CountVectorizer(stop_words="english", min_df=0.005, ngram_range=(1,1))
    count_vectorizer.fit(data.stemmed)
    transformed = count_vectorizer.transform(data.stemmed)

    tfidf_transform = TfidfTransformer()
    transformed_weights = tfidf_transform.fit_transform(transformed)

    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': count_vectorizer.get_feature_names(), 'weight': weights})
    weights_df.sort_values(by='weight', ascending=False).head(50) # return first 50 with biggest weight
    transformed_weights.toarray()

    tf_idf = pd.DataFrame(transformed_weights.todense(), index=data['index'], columns=count_vectorizer.get_feature_names())
    tf_idf.to_csv('data/tfidf_features.csv')
