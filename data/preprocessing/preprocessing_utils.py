import re
import unicodedata
import contractions
import string
import nltk
from slovene_stemmer import stem


def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zčšžđćA-ZČŠĆĐŽ0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)


def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


def stemming(text, lang):
    if lang == "slo":
        text = ' '.join(stem(text.split(' ')))
    else:
        stemmer = nltk.porter.PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()
