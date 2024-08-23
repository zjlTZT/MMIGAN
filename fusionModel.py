import tensorflow as tf
from keras import Input
from keras.layers import Softmax, LayerNormalization, Reshape, BatchNormalization
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, Conv2DTranspose, Flatten, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling2D, UpSampling1D,MultiHeadAttention

class TransformerModel(tf.keras.Model):
    def __init__(self, ):
        super(TransformerModel, self).__init__()
        self.image_input = Input(shape=(512,))
        self.text_input = Input(shape=(384,))
        num_heads = 8
        key_dim = 512
        dropout = 0.1
        self.attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.normalization_layer = LayerNormalization(epsilon=1e-6)
        self.reshape_layer = Reshape((1, -1))
        self.output_layer_image = Dense(512)
        self.output_layer_text = Dense(384)
    def call(self, inputs1,inputs2):
        combined_input = tf.concat((inputs1,inputs2),axis=1)
        combined_input_reshaped = self.reshape_layer(combined_input)
        attention_output = self.attention_layer(combined_input_reshaped, combined_input_reshaped)
        attention_output = tf.squeeze(attention_output, axis=1)
        attention_output = self.normalization_layer(attention_output)
        output_image = self.output_layer_image(attention_output)
        return output_image