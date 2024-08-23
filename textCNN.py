import numpy as np
import tensorflow as tf
from keras.layers import concatenate, BatchNormalization, Dropout
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, Conv2DTranspose, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling2D, UpSampling1D
from tensorflow.keras import Model
import utilss
class text_embedding(Model):
    def __init__(self, maxlen, max_features, embedding_dims):
        super().__init__()
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.maxlen = maxlen
    def call(self, inputs):
        emb = self.embedding(inputs)
        emb = tf.transpose(emb, [0, 1, 3, 2])
        return emb
class text_encoder(Model):
    def __init__(self, filter_num, embedding_dims, kernel_regularizer):
        super(text_encoder, self).__init__()
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(2, embedding_dims), activation='relu',
                   kernel_regularizer=kernel_regularizer)
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, embedding_dims), activation='relu',
                            kernel_regularizer=kernel_regularizer)
        self.conv3 = Conv2D(filters=filter_num, kernel_size=(4, embedding_dims), activation='relu',
                            kernel_regularizer=kernel_regularizer)
        self.pool = GlobalMaxPooling2D()
    def call(self, inputs):
        conca = []
        a1 = self.conv1(inputs)
        block1 = a1
        b1 = self.pool(a1)
        conca.append(b1)

        a2 = self.conv2(inputs)
        block2 = a2
        b2 = self.pool(a2)
        conca.append(b2)

        a3 = self.conv3(inputs)
        block3 = a3
        b3 = self.pool(a3)
        conca.append(b3)
        x = Concatenate()(conca)
        return x,block1,block2,block3
