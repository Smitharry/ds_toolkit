from keras.models import Model
from keras.layers import *
import tensorflow as tf
import numpy as np
from keras.utils import plot_model


class TextCNN():
    def __init__(self, sentence_length, embedding_size, vocab_size, n_classes,
                 n_filters, filters=(2, 3, 4, 5), trainable=True, w2v_model=None,
                 vocabulary=dict(), dense_layers=None, dropout_rate=0.2,
                 loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.__text_cnn = _TextCNN(sentence_length, embedding_size, vocab_size, vocabulary, w2v_model, trainable, filters,
                            n_filters, dense_layers, n_classes, loss, metrics, dropout_rate)

    def get_model(self):
        return self.__text_cnn.model

    def save_to_png(self, filename='model.png'):
        plot_model(self.__text_cnn.model, to_file=filename)


class _TextCNN():
    def __init__(self, sentence_length, embedding_size, vocab_size, vocabulary, w2v_model, trainable, filters,
                 n_filters, dense_layers, n_classes, loss, metrics, dropout_rate):
        self.model = self.collect_cnn(sentence_length, embedding_size, vocab_size,
                                      vocabulary, w2v_model, trainable, filters,
                                      n_filters, dense_layers, n_classes, loss,
                                      metrics, dropout_rate)

    def make_embedding(self, vocab_size, embedding_size, vocabulary, w2v_model):
        """ Make embedding matrix with either random or pretrained values"""
        embedding_matrix = np.random.random((vocab_size, embedding_size))
        if w2v_model:
            for word, index in vocabulary.items():
                if word in w2v_model.vocab:
                    embedding_matrix[index + 2] = w2v_model.wv[word]
        return embedding_matrix

    def make_embedding_layer(self, sentence_length, embedding_matrix, trainable):
        """Make embedding layer from embedding matrix"""
        embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                    output_dim=embedding_matrix.shape[1],
                                    input_length=sentence_length,
                                    weights=[embedding_matrix],
                                    trainable=trainable)
        return embedding_layer

    def make_conv_layer(self, sentence_length, embedding_layer, n_filters, filters):
        """Make convolutional branches"""
        branches = []
        for filter_ in filters:
            branch = Conv1D(filters=n_filters, kernel_size=filter_,
                            activation='relu')(embedding_layer)
            branch = MaxPooling1D(pool_size=sentence_length - filter_ + 1,
                                  padding='valid', strides=None)(branch)
            branch = Flatten()(branch)
            branches.append(branch)
        return branches

    def collect_cnn(self, sentence_length, embedding_size, vocab_size, vocabulary, w2v_model, trainable, filters,
                    n_filters, dense_layers, n_classes, loss, metrics, dropout_rate):
        """Collect a whole CNN together"""
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        embedding_matrix = self.make_embedding(vocab_size, embedding_size, vocabulary, w2v_model)
        embedding_layer = self.make_embedding_layer(sentence_length, embedding_matrix, trainable)
        embedding_layer = embedding_layer(input_layer)

        branches = self.make_conv_layer(sentence_length, embedding_layer, n_filters, filters)
        conv_layer = concatenate(branches, axis=-1)
        conv_layer = Dropout(dropout_rate)(conv_layer)

        if dense_layers:
            dense_layer = dense_layers[0](conv_layer)
            for layer in dense_layers[1:]:
                dense_layer = layer(dense_layer)
            output_layer = Dense(n_classes, activation='softmax')(dense_layer)
        else:
            output_layer = Dense(n_classes, activation='softmax')(conv_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=loss,
                      metrics=metrics)
        return model
