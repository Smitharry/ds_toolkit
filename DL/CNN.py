from keras.models import Model
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *
import tensorflow as tf
import numpy as np


class TextCNN():
    def __init__(self, vocab_size, embedding_size, sentence_length,
                 n_filters, n_classes, filters=(2, 3, 4, 5), trainable=True,
                 w2v_model=None, vocabulary=dict(), dense_layers=None, dropout_rate=0.2):
        self.__vocab_size = vocab_size
        self.__embedding_size = embedding_size
        self.__sentence_length = sentence_length
        self.__trainable = trainable
        self.__n_filters = n_filters
        self.__filters = filters
        self.__w2v_model = w2v_model
        self.__vocabulary = vocabulary
        self.__dense_layers = dense_layers
        self.__embedding_matrix = np.random.random((self.__vocab_size, self.__embedding_size))
        self.__embedding_layer = None
        self.__input_layer = None
        self.__conv_layer = None
        self.__output_layer = None
        self.__n_classes = n_classes
        self.model = self.collect_cnn()

    def alter_embedding(self):
        """ Fill embedding with pretrained values"""
        if self.__w2v_model:
            for word, index in self.__vocabulary.items():
                if word in self.__w2v_model.vocab:
                    self.__embedding_matrix[index + 2] = self.__w2v_model.wv[word]

    def make_embedding_layer(self):
        """Make embedding layer from embedding matrix"""
        self.__embedding_layer = Embedding(input_dim=self.__embedding_matrix.shape[0],
                                           output_dim=self.__embedding_matrix.shape[1],
                                           input_length=self.__sentence_length,
                                           weights=[self.__embedding_matrix],
                                           trainable=self.__trainable)

    def make_conv_layer(self):
        """Make convolutional branches"""
        branches = []
        for filter_ in self.__filters:
            branch = Conv1D(filters=self.__n_filters, kernel_size=filter_,
                            activation='relu')(self.__embedding_layer)
            branch = MaxPooling1D(pool_size=self.__sentence_length - filter_ + 1,
                                  padding='valid', strides=None)(branch)
            branch = Flatten()(branch)
            branches.append(branch)
        return branches

    def collect_cnn(self, loss='sparse_categorical_crossentropy', metrics=['accuracy'], dropout_rate=0.2):
        """Collect a whole CNN together"""
        self.alter_embedding()
        self.make_embedding_layer()
        self.__input_layer = Input(shape=(self.__sentence_length,), dtype='int32')
        self.__embedding_layer = self.__embedding_layer(self.__input_layer)

        branches = self.make_conv_layer()
        self.__conv_layer = concatenate(branches, axis=-1)
        self.__conv_layer = Dropout(dropout_rate)(self.__conv_layer)

        if self.__dense_layers:
            dense_layer = self.__dense_layers[0](self.__conv_layer)
            for layer in self.__dense_layers[1:]:
                dense_layer = layer(dense_layer)
            self.__output_layer = Dense(self.__n_classes, activation='softmax')(dense_layer)
        else:
            self.__output_layer = Dense(self.__n_classes, activation='softmax')(self.__conv_layer)

        model = Model(inputs=self.__input_layer, outputs=self.__output_layer)
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=loss,
                      metrics=metrics)
        return model
