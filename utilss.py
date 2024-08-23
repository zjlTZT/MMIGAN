import os
import jieba
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
import transform
import numpy
import numpy as np
import tensorflow as tf


class DataLoader_img():
    def __init__(self):
        t = transform.transfromm()
        self.train_data = t.train_data
        sa = np.reshape(self.train_data,(self.train_data.shape[0],self.train_data.shape[1]*self.train_data.shape[2]*self.train_data.shape[3]))
        self.num_train_data = self.train_data.shape[0]

class DataLoader_text():
    def __init__(self):
        f = open('./Data/flickr8k.txt')
        l = []
        for i in f.readlines():
            i = jieba.lcut(i, cut_all=False)
            for j in i:
                while ' ' in i:
                    i.remove(' ')
                while '.' in i:
                    i.remove('.')
                while '\n' in i:
                    i.remove('\n')
                while '\t' in i:
                    i.remove('\t')
            l.append(i)
        #########################################tokenizer编号##########################
        lengths = [len(seq) for seq in l]
        print("最小长度:", np.min(lengths))
        print("最大长度:", np.max(lengths))
        print("平均长度:", np.mean(lengths))
        print("中位数长度:", np.median(lengths))
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(l)
        sequences = tokenizer.texts_to_sequences(l)
        x_train = sequence.pad_sequences(sequences,maxlen=11, padding='post')
        ####归一化
        self.training_data = x_train
        self.max_features = len(tokenizer.word_index) +1
        self.tokenizer = tokenizer
        self.l = l


def get_batch_six(train_data1, train_data2, train_data3, train_data4, train_data5,train_data6, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        train_data1_batch = train_data1[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data2_batch = train_data2[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data3_batch = train_data3[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data4_batch = train_data4[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data5_batch = train_data5[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data6_batch = train_data6[now_batch * batch_size:(now_batch + 1) * batch_size]

        # label_batch = label[now_batch * batch_size:(now_batch + 1) * batch_size]
    else:
        train_data1_batch = train_data1[now_batch * batch_size:]
        train_data2_batch = train_data2[now_batch * batch_size:]
        train_data3_batch = train_data3[now_batch * batch_size:]
        train_data4_batch = train_data4[now_batch * batch_size:]
        train_data5_batch = train_data5[now_batch * batch_size:]
        train_data6_batch = train_data6[now_batch * batch_size:]

        # label_batch = label[now_batch * batch_size:]
    return np.array(train_data1_batch), np.array(train_data2_batch),np.array(train_data3_batch),np.array(train_data4_batch),np.array(train_data5_batch),np.array(train_data6_batch)



def normalization(data, parameters=None):
    norm_data = data
    if parameters is None:
        min_arr = tf.reduce_min(norm_data, axis=0)
        max_arr = tf.reduce_max(norm_data + 1e-6, axis=0)
        norm_data = (norm_data - min_arr)/(max_arr - min_arr)
        norm_parameters = {'min_val': min_arr,
                           'max_val': max_arr}
    else:
        min_arr = parameters['min_val']  # min by column
        max_arr = parameters['max_val']
        norm_data = (norm_data - min_arr) / (max_arr - min_arr)
        norm_parameters = parameters
    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):

    min_arr = norm_parameters['min_val']
    max_arr = norm_parameters['max_val']
    min_arr = tf.cast(min_arr, dtype=tf.float32)
    max_arr = tf.cast(max_arr, dtype=tf.float32)
    norm_data = tf.cast(norm_data, dtype=tf.float32)
    renorm_data = norm_data * (max_arr - min_arr) + min_arr
    return renorm_data


def binary_sampler(p, a, b, c,d):
    '''Sample binary random variables.

  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
    np.random.seed(0)
    unif_random_matrix = np.random.uniform(0., 1., size=[a, b,c,d])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

def miss_data_gen(miss_rate, data):
    data_m = binary_sampler(1 - miss_rate, data.shape[0],
                            data.shape[1], data.shape[2], data.shape[3]).astype(np.float32)

    miss_data = np.array(data)
    miss_data[data_m == 0] = 0
    return miss_data, data_m

def save_img(data, url):
    if not os.path.exists(url):
        os.makedirs(url)
    data = np.array(data)
    data = data.astype(np.float32)
    data = tf.convert_to_tensor(data)
    ori_list = tf.split(data, num_or_size_splits=data.shape[0], axis=0)
    id = 0
    for i in range(data.shape[0]):
        ima = tf.reshape(ori_list[i], shape=(data.shape[1],data.shape[2], 3))
        ima = tf.cast(ima, dtype=tf.uint8)
        ima = tf.image.encode_jpeg(ima)
        fwrite = tf.io.write_file(url + str(id) + ".jpeg", ima)
        id += 1

