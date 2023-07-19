# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:11:15 2019
https://blog.csdn.net/Solo95/article/details/85176688
@author: 82045
"""
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import tensorflow as tf
import h5py
from tflearn.datasets import imdb
import random
import os, time
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
from keras import regularizers
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# %%数据预处理
num_classes = 11  # 信号分类数
# DATA_PATH = '/public/home/zhb/JT/signal/signal_dataset/npy/RML2016.10a.pkl/'
DATA_PATH = '/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.test/'
trainX = np.load(DATA_PATH + 'radio11CNormTrainX.npy')
trainSnrY = np.load(DATA_PATH + 'radio11CNormTrainSnrY.npy')
trainSnr = trainSnrY[0]
trainY = trainSnrY[1]  # SnrY中[0]是信噪比，[1]是trainY,即调制类型(0~11)
testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')
testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy')
testSnr = testSnrY[0]
testY = testSnrY[1]

list_val = np.arange(len(testX))  # list_val==array([0,1,2,,...,44000])   len(testX)==44000
random.shuffle(list_val)  # 将序列的所有元素随机排序
val_percent = 0.1
valX = testX[list_val[0: int(len(testX) * val_percent)]]
valY = testY[list_val[0: int(len(testX) * val_percent)]]
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))

trainY = tflearn.data_utils.to_categorical(trainY, num_classes)
valY = tflearn.data_utils.to_categorical(valY, num_classes)
testY = tflearn.data_utils.to_categorical(testY, num_classes)
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))

trainX = np.reshape(trainX, (len(trainX), 16, 16));
valX = np.reshape(valX, (len(valX), 16, 16));
testX = np.reshape(testX, (len(testX), 16, 16));
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))

in_shp = list(trainX.shape[1:])

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    # valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    # Third component of main path (≈2 lines)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    ### END CODE HERE ###

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)


    return X


def ResNet50(input_shape=(16, 16), classes=11):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    weight_decay = 0.0005
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = Reshape((in_shp + [1]), input_shape=in_shp)(X_input)
    X = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(X)
    # X = Conv2D(64, (3, 3), padding='same')
    # X = Flatten()(X_input)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")
    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4 (≈6 lines)
    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5 (≈3 lines)
    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc" + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


# %%
model = ResNet50(input_shape=(16, 16), classes=11)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%Set up some params
epochs = 100  # number of epochs to train on
batch_size = 64  # training batch size default 1024

'''
加载的模型大小是91.20MB
目前的特征图都是基于这个参数画的
'''
filepath = '/home/NewDisk/yelinhui/explainable/Deepsig.10A/ResNet50/resnet50six.h5'  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
# log_filename = 'model_train.csv'
print(filepath)
try:
    model.load_weights(filepath)
    print("加载模型成功!继续训练模型")
except:
    print("加载模型失败!开始训练一个新模型")

#
# history = model.fit(trainX,
#                     trainY,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(valX, valY),
#                     callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
#                         keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
#                         keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
#                                                         mode='auto'),
#                         keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size,
#                                                     write_graph=True,
#                                                     write_grads=False, write_images=False, embeddings_freq=0,
#                                                     embeddings_layer_names=None,
#                                                     embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
#                         # keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
#                     ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
#
# model_name = filepath
# model.save(filepath)
# model.save_weights(filepath)

# try:
#     model.load_weights(filepath)
#     print("加载模型成功!")
# except:
#     print("加载模型失败!")
# confusion matrix

# y_test = testSnrY[1]
# y_predict = np.zeros(shape=len(testX), dtype=np.int32)
# pre_batch = 1000
# count = 0
# correct = 0
#
# index = []
# a = np.array([18, 10], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# # train = np.reshape(train, (1, 16, 16));   # 测试数据
# result = model.predict(train)
# result = np.argmax(result, axis=1)
#
# y_test = testSnrY[1]
# y_test = y_test[index]
# count = 0
# for i in result:
#     if i == y_test[count]:
#         correct = correct+1
#     count = count+1
# acc = correct/len(y_test)
# print(acc)

# y_test = testSnrY[1]
# y_predict = np.zeros(shape=len(testX), dtype=np.int32)    # 44000
# pre_batch = 1000
# iter = int(len(testX) / pre_batch)   # 44
# for i in range(0, iter):
#     y_predict[i * pre_batch: (i + 1) * pre_batch] = np.argmax(model.predict(testX[i * pre_batch: (i + 1) * pre_batch]),
#                                                               axis=1)
# import sklearn
# test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
# test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
# print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)

# np.save('/home/NewDisk/yelinhui/explainable/Deepsig.10A/vgg16/TESTconfusion.npy', (y_predict, y_test, testSnrY[0]))
# np.save('CMsix', test_CM)

'''
获取模型卷积层的输出
'''
# def conv_out(model, layer_name):
#     for l in model.layers:
#         if l.name == layer_name:
#             return l.output
#
# model_1 = Model(inputs=model.input, outputs=conv_out(model, 'res5c_branch2c'))
# # feature = K.function([model_1.input], [model_1.output])  # 计算LSTM层选中的神经元的输出之和对输入的导数
# # feature_out = feature([img])
#
# data = []
# label = []
# for c in range(11):
#     index = []
#     a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:  # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count + 1
#     index = np.array(index)  # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 16, 16));  # 测试数据
#     y_test = testSnrY[1]
#     y_test = y_test[index]
#     # result = model.predict(train)
#     # result = np.argmax(result, axis=1)
#
#     # count = 0
#     # right_index = []
#     # for k in result:
#     #     if k == a[1]:
#     #         right_index.append(count)
#     #     count = count + 1
#     #
#     # if right_index != []:
#     #     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     #     right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#     #     right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
#     # else:
#     #     right_train = []
#     #     right_y = []  # 找到的所有预测正确的数据的类标
#
#     feature = model_1.predict(train[0:80])
#     label_c = y_test[0:80]
#     label.append(label_c)
#     data.append(feature)
#
# label = np.array(label)
# label = label.reshape(-1)
# data = np.array(data)
# print(data.shape)
# data = data.reshape(-1, 2048)
# print(label.shape)
# print(data.shape)
#
# np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/TSNE/Restnet模型特征图/data.npy", data)
# np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/TSNE/Restnet模型特征图/label.npy", label)


'''
测试模型对各个信噪比的准确性
'''
# accuracy = np.zeros(20)
# c = 0
# for snr in range(-20, 20, 2):
#     index = []
#     a = np.array([snr], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引，count其实就是找到的索引，0是snr，1是信号标签
#         if i[0] == a[0]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)    # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 16, 16));   # 测试数据
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#     import sklearn
#     y_test = testSnrY[1]
#     y_test = y_test[index]
#     test_acc = sklearn.metrics.accuracy_score(y_test, result)
#     accuracy[c] = test_acc
#     c = c+1
#     print(accuracy)
# np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/识别准确率/"
#         "各个信噪比下的识别准确率/Resnet模型各个信噪比下的识别准确率.npy", accuracy)


'''
计算模型在各个信噪比下，对各个信号类别的识别准确率
'''
# import xlwt
# category = {0:"WBFM", 1:"QPSK", 2:"QAM64", 3: "QAM16", 4:"PAM4", 5:"GFSK", 6:"CPFSK" ,7:"BPSK", 8:"8PSK", 9:"AM-SSB", 10:"AM-DSB"}
# workbook = xlwt.Workbook(encoding='ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# style = xlwt.XFStyle()  # 初始化样式
# font = xlwt.Font()  # 为样式创建字体
# font.name = 'Times New Roman'
# font.bold = True  # 黑体
# font.underline = True  # 下划线
# font.italic = True  # 斜体字
# style.font = font  # 设定样式
# worksheet.write(0, 0, '信噪比')
# row = 0
# for snr in range(-20, 20, 2):
#     row = row + 1
#     worksheet.write(row, 0, snr)
#
# colum = 0
# result_save = np.zeros([20, 11])
#
# for c in range(11):
#     colum = colum + 1
#     worksheet.write(0, colum, str(category[c]))
#     row = 0
#     for snr in range(-20, 20, 2):
#         row = row+1
#         index = []
#         a = np.array([snr, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#         count = 0
#         for i in testSnrY.T:                         # 找所需要测试的信号的索引
#             if i[0] == a[0] and i[1] == a[1]:
#                 index.append(count)
#             count = count+1
#         index = np.array(index)    # 所找到的索引
#         train = testX[index]
#         train = np.reshape(train, (len(train), 16, 16));   # 测试数据
#         result = model.predict(train)
#         result = np.argmax(result, axis=1)
#
#         import sklearn
#         y_test = testSnrY[1]
#         y_test = y_test[index]
#         test_acc = sklearn.metrics.accuracy_score(y_test, result)
#         worksheet.write(row, colum, test_acc)
#         result_save[row-1, colum-1] = test_acc
#         np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/识别准确率/各个信噪比下模型对各个类的识别准确率"
#                 "/Resnet_对各个类信号的识别准确率.npy", result_save)
#         # print("对应类别的类标{}_信噪比{}_准确率{}".format(c, snr, test_acc))
#     workbook.save('./模型的识别准确率.xls')


'''
测试模型在某个信噪比下对某个类的识别准确率
'''
# index = []
# a = np.array([18, 9], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[1] == a[0]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
#
# train = testX[index]
# train = np.reshape(train, (len(train), 16, 16));   # 测试数据
# result = model.predict(train)
# result = np.argmax(result, axis=1)
# print(result)
# import sklearn
# y_test = testSnrY[1]
# y_test = y_test[index]
# print(y_test)
# test_acc = sklearn.metrics.accuracy_score(y_test, result)
# print(test_acc)
