import pandas as pd
import numpy as np

if __name__ == "__main__":
    bad_words = pd.read_csv('data/hate_words.csv', header=None)
    data = pd.read_csv('../../data/twitter/twitter_preprocessed.csv')
    tweets = data['tweet']

    # convert bad words to list
    bad_words_list = bad_words[0].tolist()
    hate_words_percentages = []
    for i in range(len(tweets)):
        count = 0
        tweet_words = str(tweets[i]).split(" ") # convert to string in case tweet is just a number or something
        for tweet_word in tweet_words:
            if tweet_word in bad_words_list:
                count += 1
        hate_words_percentages.append(round(count / len(tweet_words), 2))
    # append hate words info and save
    data["hate"] = hate_words_percentages
    data.to_csv('data/sentiment_scores.csv')
