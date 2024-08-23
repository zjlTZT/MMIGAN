import numpy as np
import torch
from keras.layers import concatenate, BatchNormalization
import fusionModel
from textCNN import text_embedding
from textCNN import text_encoder
import time
import utilss
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, MaxPool1D




class encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same',
                                   activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))


        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))


        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))


        self.conv8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv9 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv10 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))


        self.conv11 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv12 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv13 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.fla = tf.keras.layers.Flatten()
    def call(self, train_data):

        x = self.conv1(train_data)
        x = BatchNormalization()(x)
        x = self.conv2(x)
        x = BatchNormalization()(x)
        block1_out = x
        x = self.pool1(x)

        x = self.conv3(x)
        x = BatchNormalization()(x)
        x = self.conv4(x)
        x = BatchNormalization()(x)
        block2_out = x
        x = self.pool2(x)

        x = self.conv5(x)
        x = BatchNormalization()(x)
        x = self.conv6(x)
        x = BatchNormalization()(x)
        x = self.conv7(x)
        x = BatchNormalization()(x)
        block3_out = x
        x = self.pool3(x)

        x = self.conv8(x)
        x = BatchNormalization()(x)
        x = self.conv9(x)
        x = BatchNormalization()(x)
        x = self.conv10(x)
        x = BatchNormalization()(x)
        block4_out = x
        x = self.pool4(x)

        x = self.conv11(x)
        x = BatchNormalization()(x)
        x = self.conv12(x)
        x = BatchNormalization()(x)
        x = self.conv13(x)
        x = BatchNormalization()(x)
        block5_out = x
        x = self.pool5(x)

        x = self.fla(x)
        return x,block1_out,block2_out,block3_out,block4_out,block5_out
class decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv2 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv3 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )

        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv4 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv5 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv6 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', )

        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', )
        self.unconv8 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', )
        self.unconv9 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', )

        self.unpool4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv10 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', )
        self.unconv11 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', )

        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv12 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', )
        self.unconv13 = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same', )

        self.redense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.redense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.redense3 = tf.keras.layers.Dense(units=6144,activation=tf.nn.relu)
        self.redense4 = tf.keras.layers.Dense(units=8192,activation=tf.nn.relu)

    def call(self, inputs,block1_out,block2_out,block3_out,block4_out,block5_out):
        x = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 1, 1, 512))
        x = self.unpool1(x)
        x = self.unconv1(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block5_out])
        x = self.unconv2(x)
        x = BatchNormalization()(x)
        x = self.unconv3(x)
        x = BatchNormalization()(x)

        x = self.unpool2(x)
        x = self.unconv4(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block4_out])
        x = self.unconv5(x)
        x = BatchNormalization()(x)
        x = self.unconv6(x)
        x = BatchNormalization()(x)

        x = self.unpool3(x)
        x = self.unconv7(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block3_out])
        x = self.unconv8(x)
        x = BatchNormalization()(x)
        x = self.unconv9(x)
        x = BatchNormalization()(x)

        x = self.unpool4(x)
        x = self.unconv10(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block2_out])

        x = self.unconv11(x)
        x = BatchNormalization()(x)

        x = self.unpool5(x)
        x = self.unconv12(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block1_out])
        x = self.unconv13(x)
        x = BatchNormalization()(x)
        return x


class D_encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same',
                                   activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, train_data):

        x = self.conv1(train_data)
        x = BatchNormalization()(x)
        x = self.conv2(x)
        x = BatchNormalization()(x)
        block1_out = x
        x = self.pool1(x)
        x = self.fla(x)
        return x,block1_out
class D_decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv12 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', )
        self.unconv13 = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', )
    def call(self, inputs,block1_out):
        x = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 16, 16, 64))
        x = self.unpool5(x)
        x = self.unconv12(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block1_out])
        x = self.unconv13(x)
        x = BatchNormalization()(x)
        return x

class Image_CNN(torch.nn.Module):
    def __init__(self,maxlen, max_features, embedding_dims, filter_num, kernel_regularizer, train_ds_text):
        super(Image_CNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.filter_num = filter_num
        self.kernel_regularizer = kernel_regularizer
        self.train_ds_text = train_ds_text
        self.learning_rate = 0.0001
        self.encoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.decoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.text_encoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.text_decoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.bert_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.emb_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.fusion_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_en_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_de_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.img_en = encoder()
        self.img_de = decoder()
        self.Dis_en = D_encoder()
        self.Dis_de = D_decoder()

        self.embedding = text_embedding(self.maxlen, self.max_features, self.embedding_dims)
        self.text_en = text_encoder(self.filter_num, self.embedding_dims, self.kernel_regularizer)
        self.text_de = text_decoder(self.filter_num, self.embedding_dims, self.kernel_regularizer,self.maxlen)
        self.fusion = fusionModel.TransformerModel()


    def train_model_img(self,  batch_size):
        miss_rate = 0.5
        hint_rate = 0.9
        print("Start training...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        train_ds_img = utilss.DataLoader_img()
        ori_data = train_ds_img.train_data
        miss_data, data_m = utilss.miss_data_gen(miss_rate,ori_data)
        miss_rate_guancezhi = 0.3
        non_zero_indices = np.where(miss_data != 0)
        num_to_zero = int(len(non_zero_indices[0]) * miss_rate_guancezhi)
        zero_indices = np.random.choice(len(non_zero_indices[0]), num_to_zero, replace=False)
        miss_data[non_zero_indices[0][zero_indices], non_zero_indices[1][zero_indices], non_zero_indices[2][zero_indices],non_zero_indices[3][zero_indices]] = 0

        miss_data_guance = (miss_data != 0).astype(np.float32)


        data_h_temp = utilss.binary_sampler(hint_rate, ori_data.shape[0], ori_data.shape[1], ori_data.shape[2],
                                            ori_data.shape[3]).astype(np.float32)
        data_h = np.array((data_h_temp * data_m)).astype(np.float32)

        train_data = miss_data / 255.0
        total_batch = int(train_ds_img.num_train_data / batch_size)
        print("total_batch:", total_batch)

        training_epochs = 1501
        for epoch in range(training_epochs):
                id = 0
                totalLoss = 0
                for batch_index in range(total_batch):
                    batch_ori_data, batch_data_img, batch_data_text,batch_data_m,batch_data_m_guance, batch_data_h = utilss.get_batch_six(
                                                                                ori_data / 255.0,
                                                                                train_data,
                                                                                self.train_ds_text.training_data,
                                                                                data_m,
                                                                                miss_data_guance,
                                                                                data_h,
                                                                                batch_size,batch_index,total_batch)
                    batch_data_img = batch_data_img.astype(np.float32)
                    batch_data_img = tf.convert_to_tensor(batch_data_img)
                    batch_data_text = tf.reshape(batch_data_text,
                                                 shape=(batch_data_text.shape[0], batch_data_text.shape[1], 1))

                    batch_ori_data = batch_ori_data.astype(np.float32)
                    batch_ori_data = tf.convert_to_tensor(batch_ori_data)
                    id = id + 1
                    with tf.GradientTape() as img_en_tape, tf.GradientTape() as img_de_tape,\
                            tf.GradientTape() as emb_tape, tf.GradientTape() as text_en_tape, \
                            tf.GradientTape() as text_de_tape,tf.GradientTape() as dis_tape,tf.GradientTape() as fusion_tape,\
                            tf.GradientTape() as dis_en_tape,tf.GradientTape() as dis_de_tape:
                        encoder_img_res,block1,block2,block3,block4,block5 = self.img_en(batch_data_img)###vgg16没有flatten，所以不用reshape
                        emb = self.embedding(batch_data_text)
                        encoder_text_res,t_block1,t_block2,t_block3 = self.text_en(emb)
                        fusion_data = self.fusion(encoder_img_res, encoder_text_res)
                        decoder_res = self.img_de(fusion_data,block1,block2,block3,block4,block5)
                        Hat_x = batch_ori_data * batch_data_m + decoder_res * (1 - batch_data_m)
                        temp = tf.concat(values=[Hat_x, batch_data_h], axis=3)
                        D_prob_en,d_block1 = self.Dis_en(temp)
                        D_prob = self.Dis_de(D_prob_en,d_block1)

                        self.mse_loss_noguance = tf.reduce_sum(
                            ((batch_data_m-batch_data_m_guance) * batch_ori_data - (batch_data_m-batch_data_m_guance) * decoder_res) ** 2) / tf.reduce_sum(
                            (batch_data_m-batch_data_m_guance))

                        self.mse_loss_guance = tf.reduce_sum(
                            (batch_data_m_guance * batch_ori_data - batch_data_m_guance * decoder_res) ** 2) / tf.reduce_sum(
                            batch_data_m)
                        self.G_loss_temp = -tf.reduce_mean((1 - batch_data_m) * tf.math.log(D_prob + 1e-8))
                        self.loss = self.G_loss_temp + self.mse_loss_noguance * 1000 + self.mse_loss_guance *1000
                        self.D_loss_temp = -tf.reduce_mean(
                            batch_data_m * tf.math.log(D_prob + 1e-8) + (1 - batch_data_m) * tf.math.log(1. - D_prob + 1e-8))
                        imputed_loss = tf.reduce_sum(
                            ((1-batch_data_m) * batch_ori_data - (1-batch_data_m) * decoder_res) ** 2) / tf.reduce_sum(
                            1-batch_data_m)
                        totalLoss+=imputed_loss
                        self.gradients_of_en = img_en_tape.gradient(self.loss,
                                                                     self.img_en.trainable_variables)
                        self.encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_en, self.img_en.trainable_variables))
                        self.gradients_of_de = img_de_tape.gradient(self.loss,
                                                                     self.img_de.trainable_variables)
                        self.decoder_optimizer.apply_gradients(
                            zip(self.gradients_of_de, self.img_de.trainable_variables))
                        self.gradients_of_fusion = fusion_tape.gradient(self.loss,#+self.text_loss,
                                                                    self.fusion.trainable_variables)
                        self.fusion_optimizer.apply_gradients(
                            zip(self.gradients_of_fusion, self.fusion.trainable_variables))
                        self.gradients_of_text_en = text_en_tape.gradient(self.loss,
                                                                 self.text_en.trainable_variables)
                        self.text_encoder_optimizer.apply_gradients(
                           zip(self.gradients_of_text_en, self.text_en.trainable_variables))
                        for i in range(5):
                            self.gradients_of_dis_en = dis_en_tape.gradient(self.D_loss_temp,
                                                                      self.Dis_en.trainable_variables)
                            self.dis_en_optimizer.apply_gradients(
                                zip(self.gradients_of_dis_en, self.Dis_en.trainable_variables))

                            self.gradients_of_dis_de = dis_de_tape.gradient(self.D_loss_temp,
                                                                            self.Dis_de.trainable_variables)
                            self.dis_de_optimizer.apply_gradients(
                                zip(self.gradients_of_dis_de, self.Dis_de.trainable_variables))
                        if(batch_index == 0):
                            decode_data = decoder_res
                        else:
                            tmp_res = decoder_res
                            decode_data = tf.concat((decode_data, tmp_res), axis=0)
                print("Epoch:", '%04d' % epoch,"imputed_loss:","{:.9f}".format((totalLoss / total_batch)**.5),"g_loss_temp:", "{:.9f}".format(self.G_loss_temp),"guancemse=", "{:.9f}".format(self.mse_loss_guance),"noguancemse=","{:.9f}".format(self.mse_loss_noguance),"g_loss=","{:.9f}".format(self.loss), "d_loss=", "{:.9f}".format(self.D_loss_temp))#,"text_loss:","{:.9f}".format(self.text_loss))
                losses = []
                losses.append([
                    'Epoch:''%04d' % epoch,
                    'imputed_loss:''{:.9f}'.format((totalLoss / total_batch) ** .5),
                    'g_loss_temp:''{:.9f}'.format(self.G_loss_temp),
                    'guancemse=''{:.9f}'.format(self.mse_loss_guance),
                    'noguancemse=''{:.9f}'.format(self.mse_loss_noguance),
                    'g_loss=''{:.9f}'.format(self.loss),
                    'd_loss=''{:.9f}'.format(self.D_loss_temp)
                ])
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))

