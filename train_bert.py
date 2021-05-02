
import pandas as pd
import numpy as np
from models.models import BERT
from models.utils.data_loader import BERT_dataloader

if __name__ == '__main__':
    SEQ_LENGTH = 128

    path = ['data/twitter/twitter_preprocessed.csv', 'data/twitter_slo/slo_twitter_binary_preprocessed.csv']
    tcn = ['tweet', 'tweet']
    lcn = ['is_offensive', 'sporni_govor']
    
    bert_dataloader = BERT_dataloader(seq_length=SEQ_LENGTH, batch_size=32)
    bert_dataloader.load_data(path, text_column_name=tcn, label_column_name=lcn, val_df=1)
    train, val = bert_dataloader.get_data()
    bert = BERT(seq_length=SEQ_LENGTH)
    model = bert.get_model()

    callbacks = bert.get_callbacks('/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/', 'twitter_ENG+SLO')
    model.fit(train, validation_data=val, epochs=20, callbacks=callbacks)
