import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
from random import sample
from sklearn.model_selection import train_test_split 

class BERT_dataloader():
    def __init__(self, seq_length=256, batch_size=32, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.seq_length = seq_length
        self.batch_size = batch_size

    def load_data(self, path, text_column_name, label_column_name, val_df=0):
        if type(path) == list:
            df = pd.read_csv(path[0])
            df = df.dropna()
            labels = df[label_column_name[0]]
            texts = df[text_column_name[0]]
            for i in range(1, len(path)):
                df = pd.read_csv(path[i])
                df = df.dropna()

                if val_df == i:
                    df, val_df = train_test_split(df, test_size=0.1)

                labels_ = df[label_column_name[i]]
                texts_ = df[text_column_name[i]]
                texts = pd.concat([texts, texts_], ignore_index=True)
                labels = pd.concat([labels, labels_], ignore_index=True)

                val_labels = val_df[label_column_name[i]]
                val_texts = val_df[text_column_name[i]]

        else:
            df = pd.read_csv(path)
            df = df.dropna()
            df, val_df = train_test_split(df, test_size=0.2)
            labels = df[label_column_name]
            texts = df[text_column_name]

            val_labels = val_df[label_column_name]
            val_texts = val_df[text_column_name]

        labels = (labels.values[:,None] != np.arange(2)).astype(int)
        val_labels = (val_labels.values[:,None] != np.arange(2)).astype(int)

        x1 = np.zeros((len(labels), self.seq_length))
        x2 = np.zeros((len(labels), self.seq_length))

        for i, text in enumerate(texts):
            ids, mask = self.tokenize(text)
            x1[i, :] = ids
            x2[i, :] = mask

        val_x1 = np.zeros((len(val_labels), self.seq_length))
        val_x2 = np.zeros((len(val_labels), self.seq_length))

        for i, text in enumerate(val_texts):
            ids, mask = self.tokenize(text)
            val_x1[i, :] = ids
            val_x2[i, :] = mask

        self.train_data = tf.data.Dataset.from_tensor_slices((x1, x2, labels))
        self.train_data = self.train_data.map(self.map_func)
        self.train_data = self.train_data.shuffle(42).batch(self.batch_size)

        self.val_data = tf.data.Dataset.from_tensor_slices((val_x1, val_x2, val_labels))
        self.val_data = self.val_data.map(self.map_func)
        self.val_data = self.val_data.shuffle(42).batch(self.batch_size)

    def get_data(self):
        return self.train_data, self.val_data

                
    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(text, max_length=self.seq_length, truncation=True, padding='max_length', add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False, return_tensors='tf')

        return tokens['input_ids'], tokens['attention_mask']

    def map_func(self, input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels