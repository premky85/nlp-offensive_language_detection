import re
import unicodedata
import contractions
import string
import nltk
# from slovene_stemmer import stem

stemmer = nltk.stem.SnowballStemmer('english')

def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zčšžđćA-ZČŠĆĐŽ0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)


def remove_punctuation(text):
    # text = ' '.join([c for c in text if c not in string.punctuation])
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    return text


def stemming(text, lang):
    if lang == "slo":
        text = ' '.join(stem(text.split(' ')))
    else:
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


def expand_contractions(text):
    processed_text = ""
    for word in text.split(' '):
        processed_text += contractions.fix(word) + " "
    return processed_text


def remove_numbers(text):
    return re.sub(r'[0-9]+', '', text)


def remove_links(text):
    return re.sub(r"http\S+", "", text)


def clean_text(text, params, lang, print_text=False):
    if print_text:
        print(text)

    if "lower" in params:
        text = text.lower()
    if "accented" in params:
        text = remove_accented_chars(text)
    if "links" in params:
        text = remove_links(text)
    if "special" in params:
        text = remove_special_characters(text)
    if "contractions":
        text = expand_contractions(text)
    if "punct":
        text = remove_punctuation(text)
    if "numbers" in params:
        text = remove_numbers(text)
    if "whitespaces":
        text = remove_extra_whitespace_tabs(text)
    if "stemming":
        text = stemming(text, lang)

    if print_text:
        print(text)

    return text
