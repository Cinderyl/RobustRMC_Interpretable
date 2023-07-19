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
from keras.layers.core import Lambda
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape, Dropout
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
import keras.backend as K

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

list_val = np.arange(len(testX))  # list_val==array([0,1,2,,...,155999])   len(testX)==156000
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
'''
(176000, 16, 16) (176000, 11) (4400, 16, 16) (4400, 11) (44000, 16, 16) (44000, 11)
'''

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

    ### END CODE HERE ###

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

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    # X = Dense(1024)(X)

    X = Dense(64 * 64 * 3)(X)
    X = Dropout(0.5)(X)
    X = Reshape([64, 64, 3])(X)

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


model = ResNet50(input_shape=(16, 16), classes=11)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 100  # number of epochs to train on
batch_size = 64  # training batch size default 1024

"加载的模型的大小为307.45MB"
filepath = '/home/NewDisk/yelinhui/explainable/Deepsig.10A/ResNet50/resnet50best.h5'  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
log_filename = 'model_train.csv'
# try:
#     model.load_weights(filepath)
#     print("加载模型成功!继续训练模型")
# except:
#     print("加载模型失败!开始训练一个新模型")
#
# history = model.fit(trainX,
#                     trainY,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(valX, valY),
#                     callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
#                         keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
#                         keras.callbacks.ModelCheckpoint('resnet50best.h5', monitor='val_acc', verbose=0,
#                                                         save_best_only=True, mode='auto')
#
#                         # ,keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
#                         #                            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                         #                           embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
#
#                         # keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
#
#                     ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
#
# model_name = filepath
# model.save(filepath)
# model.save_weights(filepath)

try:
    model.load_weights(filepath)
    print("加载模型成功!")
except:
    print("加载模型失败!")


# y_test = testSnrY[1]
# y_predict = np.zeros(shape=len(testX), dtype=np.int32)
# pre_batch = 1000
# iter = int(len(testX) / pre_batch)
# for i in range(0, iter):
#     y_predict[i * pre_batch: (i + 1) * pre_batch] = np.argmax(model.predict(testX[i * pre_batch: (i + 1) * pre_batch]),
#                                                               axis=1)
# import sklearn
# test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
# test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
# print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)
# np.save("./CM.npy", test_CM)

import keras.backend as K
import tensorflow as tf
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

'''
计算var_list对tensor的梯度
'''
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

def conv_out(model, layer_name):
    for l in model.layers:
        if l.name == layer_name:
            return l.output

def grad_cam(input_model, img, category_index, layer_name):
    global conv_output
    ori_img = img
    nb_classes = num_classes
    def target_layer(x): return target_category_loss(  # 好像是定义的交叉熵
        x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    # model.summary()
    loss = K.sum(model.output)
    # 遍历各层，当layer_name=="block5_conv3"时 即得到layer_name对应的输出conv_output，
    # conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    '''
    for l in model.layers:
        #print (l)
        if l.name == layer_name:
            conv_output = l.output
    '''
    conv_output = conv_out(model, layer_name)
    # 获取卷积输出对loss的梯度矩阵,并作normalize
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([img])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    # 每个滤波器的梯度求和并取均值获得各滤波器对应权重
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)


    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    _, width, height = ori_img.shape    # 16x16

    height = 540
    width = 540

    # height = 1080
    # width = 360

    # cam = cam.reshape(16,1)
    # cam = cv2.resize(cam, (16, 16))
    # cam = cam.reshape(128, 2)

    cam_weight = cv2.resize(cam, (1, 128))

    # cam = cv2.resize(cam, (1, 128))
    cam = cv2.resize(cam, (width, height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam)
    cam = 255 * cam / np.max(cam)
    print("success")
    return np.uint8(cam), cam_weight

'''
单张画图
'''
dic = {0: "WBFM", 1: "QPSK", 2: "QAM64", 3: "QAM16", 4: "PAM4",5: "GFSK", 6: "CPFSK", 7: "BPSK",
       8: "8PSK", 9: "AM-SSB", 10: "AM-DSB"}
index_dic = {0: 15, 1: 0, 2: 16, 3: 17, 4: 11, 5: 19, 6: 6, 7: 0, 8: 17, 9: 8, 10: 0}
import numpy as np
d = 0
index = index_dic[d]
path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/LSTM128_128/label_{}.npy".format(d)
label = d
img = np.load(path)[index].reshape(1, 16, 16)
cam, cam_wight = grad_cam(model, img, label, 'conv2d_9')
radio11CamweightTestX = cam_wight.reshape(128)

if min(radio11CamweightTestX) < 0:
    radio11CamweightTestX = abs(min(radio11CamweightTestX)) + radio11CamweightTestX
    radio11CamweightTestX = radio11CamweightTestX / max(radio11CamweightTestX)
else:
    radio11CamweightTestX = radio11CamweightTestX / max(radio11CamweightTestX)

for i in range(128):
    if radio11CamweightTestX[i] < 0.90:
        radio11CamweightTestX[i] = 0
    # else:
    #     radio11CamweightTestX[i] = 0.9

'''
正确的星座图
'''
plt.figure(figsize=(7, 7))
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid(linestyle='-.')
signal = img.reshape(128, 2)
first_a = signal[0:128, 0]
last_a = signal[0:128, 1]
plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=radio11CamweightTestX, s=100)
# plt.colorbar()
outputfile_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/放论文里的图/实验部分的可视化图/" \
                  "4.3节可视化分类器/4.3.1对不同分类器的可视化/VGG16/label{}_{}.png".format(d, dic[d])
plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
plt.show()
plt.close()
