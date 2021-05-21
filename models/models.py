import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoConfig, TFAutoModelForSequenceClassification

class TextClassificationNet():
    def __init__(self, seq_length=256, language_model='bert', *args, **kwargs):
        self.model_names = {'bert': 'bert-base-multilingual-cased', 'xlmr': 'xlm-roberta-large', 'csebert': 'EMBEDDIA/crosloengual-bert'}
        self.language_model = language_model
        self.bert = TFAutoModel.from_config(AutoConfig.from_pretrained(self.model_names[self.language_model]))
        self.seq_length = seq_length

    def get_model(self):
        input_ids = tf.keras.layers.Input(shape=(self.seq_length,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.seq_length,), name='attention_mask', dtype='int32')

        embeddings = self.bert(input_ids, attention_mask=mask)[0]
        X = layers.Conv1D(64, 2, padding='valid', activation='relu')(embeddings)
        X = layers.Conv1D(64, 3, padding='valid', activation='relu')(X)
        X = layers.Conv1D(64, 4, padding='valid', activation='relu')(X)

        X = layers.GlobalMaxPool1D()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dense(128, activation='relu')(X)
        X = layers.Dropout(0.1)(X)
        y = layers.Dense(4, activation='softmax', name='outputs')(X)

        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

        model.layers[2].trainable = False

        optimizer = tf.keras.optimizers.Adam(0.001)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

        return model

    def get_callbacks(self, filepath, dataset_name):
        filepath = filepath + '{}_{}_'.format(self.language_model, dataset_name) + '{epoch:04d}_CA_{val_accuracy:05.4f}.h5'

        cp_callback_0 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
        )

        return [cp_callback_0]
    
    def get_model_name(self):
        return self.model_names[self.language_model]
