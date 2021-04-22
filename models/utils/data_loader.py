import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf

class BERT_dataloader():
    def __init__(self, seq_length=256, batch_size=32, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.seq_length = seq_length
        self.batch_size = batch_size

    def get_dataloader(self, path, text_column_name, label_column_name):
        df = pd.read_csv(path)
        df = df.dropna()
        labels = df[label_column_name].values
        labels = (labels[:,None] != np.arange(2)).astype(int)
        #print(df[text_column_name].map(len).max())

        x1 = np.zeros((len(labels), self.seq_length))
        x2 = np.zeros((len(labels), self.seq_length))

        for i, text in enumerate(df[text_column_name]):
            ids, mask = self.tokenize(text)
            x1[i, :] = ids
            x2[i, :] = mask

        dataset = tf.data.Dataset.from_tensor_slices((x1, x2, labels))
        dataset = dataset.map(self.map_func)
        dataset = dataset.shuffle(42).batch(self.batch_size)
        train = dataset.take(round(len(list(dataset)) * 0.9))
        val = dataset.skip(round(len(list(dataset)) * 0.9))

        return train, val

    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(text, max_length=self.seq_length, truncation=True, padding='max_length', add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False, return_tensors='tf')

        return tokens['input_ids'], tokens['attention_mask']

    def map_func(self, input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels