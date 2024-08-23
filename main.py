import imageCNN_fusion_unet_all
import utilss

text_data = utilss.DataLoader_text()
maxlen = 11
max_features =text_data.max_features
embedding_dims = 100
filter_num = 128
kernel_regularizer = None
batch_size = 96
model_img = imageCNN_fusion_unet_all.Image_CNN(maxlen, max_features, embedding_dims, filter_num, kernel_regularizer, text_data)
model_img.train_model_img(batch_size)
