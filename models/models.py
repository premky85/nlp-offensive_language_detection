import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

class BERT():
    def __init__(self, seq_length=256, *args, **kwargs):
        self.bert = TFAutoModel.from_pretrained('bert-base-multilingual-cased')
        self.seq_length = seq_length

    def get_model(self):
        input_ids = tf.keras.layers.Input(shape=(self.seq_length,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.seq_length,), name='attention_mask', dtype='int32')

        embeddings = self.bert(input_ids, attention_mask=mask)[0]
        X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.1)(X)
        y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)

        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

        model.layers[2].trainable = False

        optimizer = tf.keras.optimizers.Adam(0.001)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

        return model

    def get_callbacks(self, filepath, dataset_name):
        filepath = filepath + 'BERT_{}_'.format(dataset_name) + '{epoch:04d}_CA_{val_accuracy:05.4f}.h5'

        cp_callback_0 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
        )

        return [cp_callback_0]
