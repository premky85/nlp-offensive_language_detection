import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
from random import sample
from sklearn.model_selection import train_test_split 

class TextDataloader():
    def __init__(self, seq_length=256, batch_size=32, language_model='bert', *args, **kwargs):
        self.model_names = {'bert': 'bert-base-multilingual-cased', 'xlmr': 'xlm-roberta-large', 'csebert': 'EMBEDDIA/crosloengual-bert'}
        self.language_model = language_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_names[self.language_model])
        self.seq_length = seq_length
        self.batch_size = batch_size

    def load_data(self, path, text_column_name, label_column_name):
        if type(path) == list:
            df = pd.read_csv(path[0])
            df = df.dropna()

            N = int((df[label_column_name[0]].values == 0).sum() * 0.4)
            df = df.drop(df[df[label_column_name[0]].eq(0)].sample(N).index)

            labels = df[label_column_name[0]]
            texts = df[text_column_name[0]]
            
            for i in range(1, len(path)):
                df = pd.read_csv(path[i])
                df = df.dropna()

                N = int((df[label_column_name[i]].values == 0).sum() * 0.4)
                df = df.drop(df[df[label_column_name[i]].eq(0)].sample(N).index)

                labels_ = df[label_column_name[i]]
                texts_ = df[text_column_name[i]]
                texts = pd.concat([texts, texts_], ignore_index=True)
                labels = pd.concat([labels, labels_], ignore_index=True)

        else:
            df = pd.read_csv(path)
            df = df.dropna()
            labels = df[label_column_name]
            texts = df[text_column_name]


        texts, self.test_texts, labels, self.test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
        texts, val_texts, labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

        
        n_classes = max(labels.values) + 1
        labels = (labels.values[:,None] == np.arange(n_classes)).astype(int)
        val_labels = (val_labels.values[:,None] == np.arange(n_classes)).astype(int)

        x1 = np.zeros((len(labels), self.seq_length))
        x2 = np.zeros((len(labels), self.seq_length))

        for i, text in enumerate(texts):
            ids, mask = self.tokenize('[CLS] ' + text + ' [SEP]')
            x1[i, :] = ids
            x2[i, :] = mask

        val_x1 = np.zeros((len(val_labels), self.seq_length))
        val_x2 = np.zeros((len(val_labels), self.seq_length))

        for i, text in enumerate(val_texts):
            ids, mask = self.tokenize('[CLS] ' + text + ' [SEP]')
            val_x1[i, :] = ids
            val_x2[i, :] = mask

        self.train_data = tf.data.Dataset.from_tensor_slices((x1, x2, labels))
        self.train_data = self.train_data.repeat()
        self.train_data = self.train_data.shuffle(42).batch(self.batch_size)
        self.train_data = self.train_data.map(self.map_func)

        self.val_data = tf.data.Dataset.from_tensor_slices((val_x1, val_x2, val_labels))
        self.val_data = self.val_data.shuffle(42).batch(self.batch_size)
        self.val_data = self.val_data.map(self.map_func)

    def get_data(self):
        return self.train_data, self.val_data

    def get_test_data(self):
        df = pd.concat([pd.Series(self.test_texts, name='text'), pd.Series(self.test_labels, name='class')], axis=1)
        return df

    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(text, max_length=self.seq_length, truncation=True, padding='max_length', add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False, return_tensors='tf')

        return tokens['input_ids'], tokens['attention_mask']

    def map_func(self, input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels
