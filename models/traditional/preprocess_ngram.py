from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    data = pd.read_csv('../../data/twitter/twitter_preprocessed.csv')

    snow_stemmer = SnowballStemmer("english") # language is required
    data['stemmed'] = data.tweet.map(lambda tweet: ' '.join([snow_stemmer.stem(word) for word in str(tweet).split(' ')]))

    count_vectorizer = CountVectorizer(stop_words="english", min_df=0.005, ngram_range=(2,2)) # 2,2 means only bigrams
    count_vectorizer.fit(data.stemmed) # create dictionary of tokens
    transformed = count_vectorizer.transform(data.stemmed) # transform to d-t matrix

    # generate data frame from n-grams
    bigrams = pd.DataFrame(transformed.todense(), index=data['index'], columns=count_vectorizer.get_feature_names())
    bigrams.to_csv('data/bigram_features.csv')
