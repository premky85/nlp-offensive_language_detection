import pandas as pd
import ast
import re
from preprocessing_utils import *
import tweepy


def pre_clean(text):
    cleaned = re.sub(
        "(^!+)|([@][^\\s]*[:]?)|(&#\\d+;)|(https?[^\\s]*)|(&amp;)|(RT)|(&lt;)",
        "",
        text
    )
    return cleaned


def read_file_aux(file_id):
    with open(data_folder + 'all_files/' + file_id + ".txt") as file:
        text = file.read()
    return text


save_folder = "../preprocessed_data/"
data_folder = "../raw/"


# WHITE SUPREMACIST FORUM DATA PREPROCESSING
print("White supremacist forum data preprocessing...")

folder = data_folder + 'all_files/'
annotations_path = data_folder + 'annotations_metadata.csv'

df = pd.read_csv(annotations_path)
df.drop(['user_id', 'num_contexts', 'subforum_id'], axis=1, inplace=True)
df['file_id'] = df['file_id'].apply(read_file_aux)
df['label'] = df['label'].apply(lambda s: int(s == "hate"))
df.rename(columns={'file_id': 'text', 'label': 'class'}, inplace=True)
df['class'].replace({1: 2}, inplace=True)
params = ["lower", "accented", "links", "special", "contractions", "punct", "numbers", "whitespaces", "stemming"]
df['text'] = df['text'].apply(lambda x: clean_text(x, params, "eng", False))
df = df[df['text'] != ""]
df.to_csv(save_folder + 'white_supremacist_forum_preprocessed.csv')


# GAB DATA PREPROCESSING
print("Gab data preprocessing...")
csv_path = data_folder + 'gab-q.csv'

df = pd.read_csv(csv_path)
df.drop(['id'], axis=1, inplace=True)
df['hate_speech_idx'] = df['hate_speech_idx'].fillna(0)

new_df = pd.DataFrame(columns=['text', 'class'])
params = ["lower", "accented", "links", "special", "contractions", "punct", "numbers", "whitespaces", "stemming"]

for row_index, row in df.iterrows():
    hate_speech_index = row['hate_speech_idx']
    sent = row['text'].split('\n')
    sent_preprocessed = []

    for s in sent:
        s = clean_text(s, params, "eng", False)
        sent_preprocessed.append(s)

    if hate_speech_index == 0:
        for s in sent_preprocessed:
            if s and s != "remov" and s != "delet":
                new_df.loc[len(new_df.index)] = [s, 0]
    else:
        hate_text_indexes = ast.literal_eval(row['hate_speech_idx'])
        for s_index, s in enumerate(sent_preprocessed, 1):
            if s and s != "remov" and s != "delet":
                if s_index in hate_text_indexes:
                    new_df.loc[len(new_df.index)] = [s, 2]
                else:
                    new_df.loc[len(new_df.index)] = [s, 0]

new_df.to_csv(save_folder + 'gab_preprocessed.csv')


# REDDIT DATA PREPROCESSING
print("Reddit data preprocessing...")
csv_path = data_folder + 'reddit-q.csv'

df = pd.read_csv(csv_path)
df.drop(['id'], axis=1, inplace=True)
df['hate_speech_idx'] = df['hate_speech_idx'].fillna(0)

new_df = pd.DataFrame(columns=['text', 'class'])
params = ["lower", "accented", "punctuation", "special", "contractions", "punct", "numbers", "whitespaces", "stemming"]

for row_index, row in df.iterrows():
    hate_speech_index = row['hate_speech_idx']
    sent = row['text'].split('\n')
    sent_preprocessed = []

    for s in sent:
        s = clean_text(s, params, "eng", False)
        sent_preprocessed.append(s)

    if hate_speech_index == 0:
        for s in sent_preprocessed:
            if s and s != "remov" and s != "delet":
                new_df.loc[len(new_df.index)] = [s, 0]
    else:
        hate_text_indexes = ast.literal_eval(row['hate_speech_idx'])
        for s_index, s in enumerate(sent_preprocessed, 1):
            if s and s != "remov" and s != "delet":
                if s_index in hate_text_indexes:
                    new_df.loc[len(new_df.index)] = [s, 2]
                else:
                    new_df.loc[len(new_df.index)] = [s, 0]

new_df.to_csv(save_folder + 'reddit_preprocessed.csv')


# FOX NEWS COMMENTS (leave in repo) DATA PREPROCESSING
print("Fox news comments preprocesing...")

df = pd.read_csv(data_folder + 'fox_news_comments.csv')
df.drop(['Unnamed: 0', 'title', 'succ', 'meta', 'user', 'mentions', 'prev'], axis=1, inplace=True)
df.rename(columns={'label': 'class'}, inplace=True)
df['class'].replace({1: 2}, inplace=True)
params = ["lower", "accented", "links", "special", "contractions", "punct", "numbers", "whitespaces", "stemming"]
df['text'] = df['text'].apply(lambda x: clean_text(x, params, "eng", False))
df = df[df['text'] != ""]
df.to_csv(save_folder + 'fox_news_comments_preprocessed.csv')


# ENG TWITTER DATA PREPROCESSING
print("English twitter data preprocessing...")

df = pd.read_csv(data_folder + 'labeled_data.csv')
df.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1, inplace=True)
df.rename(columns={'tweet': 'text'}, inplace=True)
df['class'].replace({0:2, 2:0}, inplace=True)
clean_params = ["lower", "accented", "punctuation", "special", "contractions", "punct", "numbers", "whitespaces", "stemming"]
df['text'] = df['text'].apply(pre_clean)
df['text'] = df['text'].apply(lambda x: clean_text(x, clean_params, "eng", False))
df.to_csv(save_folder + 'twitter_preprocessed.csv')


# SLO TWITTER DATA PREPROCESSING
print("Downloading Slo tweets...")

oauth_handler1 = ''
oauth_handler2 = ''
token1 = ''
token2 = ''

auth = tweepy.OAuthHandler(oauth_handler1, oauth_handler2)
auth.set_access_token(token1, token2)

api = tweepy.API(auth)

df = pd.read_csv(data_folder + 'IMSyPP_SI_anotacije_evaluation-clarin.csv', header=0)

tweets = []
labels = []

for i, tid in df[['ID', 'vrsta']].iterrows():
    try:
        tweet = api.get_status(tid['ID'], tweet_mode='extended')
        tweets.append(tweet.full_text)
        labels.append(tid['vrsta'])
    except:
        continue

print("Slo tweets data preprocessing...")

new_df = pd.DataFrame(zip(tweets, labels), columns=['text', 'class'])
new_df['class'].replace({"0 ni sporni govor": 0,
                         "2 žalitev": 1,
                         "3 nasilje": 2,
                         "1 nespodobni govor": 0}, inplace=True)

params = ["lower", "special", "punct", "whitespaces", "stemming"]
new_df['text'] = new_df['text'].apply(pre_clean)
new_df['text'] = new_df['text'].apply(lambda x: clean_text(x, params, "slo", False))
new_df.to_csv(save_folder + 'slo_twitter_preprocessed.csv')
