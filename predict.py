from operator import gt
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Model
from models.models import TextClassificationNet
from models.utils.data_loader import TextDataloader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import tensorflow as tf

if __name__ == '__main__':
    SEQ_LENGTH = 128

    base = 'data/preprocessed_data/'

    # path = [base + 'fox_news_comments.csv', base + 'gab_preprocessed.csv', base + 'reddit_preprocessed.csv', base + 'white_supremacist_forum_preprocessed.csv', base + 'twitter_preprocessed.csv', base + 'slo_twitter_binary_preprocessed.csv']
    # tcn = ['text', 'text', 'text', 'text', 'tweet', 'tweet']
    # lcn = ['label', 'class', 'class', 'is_hate', 'class', 'class_multi']

    # path = [base + 'slo_twitter_binary_preprocessed.csv']
    # tcn = ['tweet']
    # lcn = ['class_multi']

    # dataloader = TextDataloader(seq_length=SEQ_LENGTH, batch_size=32)
    # dataloader.load_data(path, text_column_name=tcn, label_column_name=lcn)
    df = pd.read_csv(base + 'slo_test_data_preprocessed.csv') # dataloader.get_test_data() # 

    labels = df['class']

    bs = 64

    model_data_eng_slo = [
        {'name': 'bert', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/bert_ENG+SLO_0016_CA_0.7948.h5', 'preds': []},
        {'name': 'csebert', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/csebert_ENG+SLO_0020_CA_0.8076.h5', 'preds': []},
        {'name': 'xlmr', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/xlmr_ENG+SLO_0020_CA_0.6831.h5', 'preds': []},
    ]

    model_data_slo = [
        {'name': 'bert', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/bert_SLO_0088_CA_0.6975.h5', 'preds': []},
        {'name': 'csebert', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/csebert_SLO_0044_CA_0.7009.h5', 'preds': []},
        {'name': 'xlmr', 'weights': '/share/Disk_2/FRI/NLP/nlp-offensive_language_detection/weights/xlmr_SLO_0015_CA_0.6195.h5', 'preds': []},
    ]

    model_data = model_data_eng_slo


    for md in model_data:
        text_dataloader = TextDataloader(seq_length=SEQ_LENGTH, batch_size=32, language_model=md['name'])
        tcn = TextClassificationNet(seq_length=SEQ_LENGTH, language_model=md['name'])
        model = tcn.get_model()
        model.load_weights(md['weights'])

        with tqdm(total=df.shape[0] // bs) as pbar:
            for i,rows in df.groupby(np.arange(len(df))//bs):
                pbar.update(1)
                tokens, masks = zip(*map(text_dataloader.tokenize, rows['text'].to_numpy()))
                if len(masks) > 1 and len(tokens) > 1:
                    tokens, masks = tf.squeeze(tf.stack(tokens)), tf.squeeze(tf.stack(masks))

                predicted = model.predict([tokens, masks])
                md['preds'] += list(predicted)

    gts = labels.to_numpy()

    for md in model_data:
        preds = np.array(md['preds']).argmax(axis=-1)
        print('=================== ' + md['name'] + ' ===================\n')
        print(confusion_matrix(gts, preds))
        print(classification_report(gts, preds, digits=3))
        print('\n\n')


    preds = (np.sum([md['preds'] for md in model_data], axis=0) / 3).argmax(axis=-1)
    print('=================== GROUP ===================\n')
    print(confusion_matrix(gts, preds))
    print(classification_report(gts, preds, digits=3))

    


    

