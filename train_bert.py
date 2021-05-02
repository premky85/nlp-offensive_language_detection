
import pandas as pd
import numpy as np
from models.models import BERT
from models.utils.data_loader import BERT_dataloader

if __name__ == '__main__':
    SEQ_LENGTH = 128
    
    path = 'data/preprocessed_data/twitter_preprocessed.csv'
    tcn = 'tweet'
    lcn = 'is_offensive'

    bert_dataloader = BERT_dataloader(seq_length=SEQ_LENGTH)
    train, val = bert_dataloader.get_dataloader(path, text_column_name=tcn, label_column_name=lcn)
    bert = BERT(seq_length=SEQ_LENGTH)
    model = bert.get_model()

    model.fit(train, validation_data=val, epochs=20)