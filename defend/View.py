# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:57:53 2019

@author: 82045
"""
import matplotlib.pyplot as plt
plt.ion()
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, Model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import h5py
import random
import time
import keras
import tflearn
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, MaxPool2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape, Dropout
import xlrd

category = {0:"WBFM", 1:"QPSK", 2:"QAM64", 3: "QAM16", 4:"PAM4", 5:"GFSK", 6:"CPFSK" ,7:"BPSK", 8:"8PSK", 9:"AM-SSB", 10:"AM-DSB"}

num_classes = 11  # 信号分类数
# DATA_PATH = '/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.test/'
DATA_PATH = '/data0/benke/ldx/explainable/dataset/'
trainX = np.load(DATA_PATH + 'read_dataradio11CNormTrainX.npy')
trainSnrY = np.load(DATA_PATH + 'read_dataradio11CNormTrainSnrY.npy')
trainSnr = trainSnrY[0]
trainY = trainSnrY[1]  # SnrY中[0]是信噪比，[1]是trainY,即调制类型(0~11)
testX = np.load(DATA_PATH + 'read_dataradio11CNormTestX.npy')
testSnrY = np.load(DATA_PATH + 'read_dataradio11CNormTestSnrY.npy')
testSnr = testSnrY[0]
testY = testSnrY[1]

# list_val==array([0,1,2,,...,155999])   len(testX)==156000
list_val =  np.arange(len(testX))
random.shuffle(list_val)  # 将序列的所有元素随机排序
val_percent = 0.1
valX = testX[list_val[0: int(len(testX) * val_percent)]]
valY = testY[list_val[0: int(len(testX) * val_percent)]]

trainY = tflearn.data_utils.to_categorical(trainY, num_classes)
valY = tflearn.data_utils.to_categorical(valY, num_classes)
testY = tflearn.data_utils.to_categorical(testY, num_classes)

trainX = np.reshape(trainX, (len(trainX), 16, 16))
valX = np.reshape(valX, (len(valX), 16, 16))
testX = np.reshape(testX, (len(testX), 16, 16))

in_shp = list(trainX.shape[1:])
weight_decay = 0.0001  # 新增
dropout = 0.5


def build_model():
    model = Sequential()
    model.add(Reshape((in_shp + [1]), input_shape=in_shp))
    model.add(Conv2D(192, (5, 5), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(160, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(96, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (5, 5), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (3, 3), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(num_classes, (1, 1), padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    sgd = tf.keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
filepath = '/data0/benke/ldx/explainable/DATA.MAT/NIN/NIN.h5'
try:
    model.load_weights(filepath)
    print("加载模型成功!")
except:
    print("加载模型失败!")

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

    cam_weight = cv2.resize(cam, (1, 128))

    # cam = cv2.resize(cam, (1, 128))
    cam = cv2.resize(cam, (width, height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam)
    cam = 255 * cam / np.max(cam)
    print("success")
    return np.uint8(cam), np.uint8(cam_weight)


'''
循环画图  画对抗样本的grad-cam
'''
# path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/Experimental Result/NIN" \
#        "模型/NIN模型上用FGSM进行目标攻击生成的对抗信号/无目标攻击/限制噪声幅值0.1/"
path ='/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_NIN/un_target_attack/'


for i in range(11):
    filepath = path+"label_{}".format(i)+"/"
    adversarial_path = filepath + "label_{}_ad_siganl_save.npy".format(i)
    source_signal = filepath+"label_{}_source_signal.npy".format(i)
    excel_path = filepath + "原类标为{}.xls".format(i)   # 存储的excel文件，这个文件里有无目标攻击的识别结果

    adversarial_signal = np.load(adversarial_path)    # 97 16 16
    source_signal = np.load(source_signal)            # 97 16 16

    baseoutput = "/data0/benke/ldx/explainable/result" \
                 "NIN模型/星座图/对抗信号的星座图/无目标攻击/限制幅值/label_{}".format(i)
    # baseoutput = "/data0/benke/ldx/explainable/result" \
    #              "NIN模型/星座图/对抗信号的星座图/无目标攻击/限制幅值/label_{}".format(i)

    if os.path.exists(baseoutput):   # 判断存储原信号的文件是否存在
        pass
    else:                                   # 若不存在，则创建这个文件夹
        os.makedirs(baseoutput)

    a_r_path = baseoutput+"/" + "对抗样本的识别结果类标的星座图"
    if os.path.exists(a_r_path):  # 判断存储原信号的文件是否存在
        pass
    else:  # 若不存在，则创建这个文件夹
        os.makedirs(a_r_path)

    a_s_path = baseoutput+"/"+"对抗样本的原类标的星座图"
    if os.path.exists(a_s_path):  # 判断存储原信号的文件是否存在
        pass
    else:  # 若不存在，则创建这个文件夹
        os.makedirs(a_s_path)

    s_s_path = baseoutput+"/"+"原信号的原类标的星座图"
    if os.path.exists(s_s_path):        # 判断存储原信号的文件是否存在
        pass
    else:  # 若不存在，则创建这个文件夹
        os.makedirs(s_s_path)

    data = xlrd.open_workbook(excel_path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    values = []
    for x in range(1, nrows):
        row = table.row_values(x)
        values.append(row)
    values = np.array(values)
    count = 0
    for img in adversarial_signal:  # 画对抗样本的识别结果类标的gradcam
        img = img.reshape(1, 16, 16)
        output_path = a_r_path
        cam, cam_weight = grad_cam(model, img, int(float(values[count, 1])), 'conv2d_9')
        radio11CamweightTestX = cam_weight
        possibilities = model.predict(img.reshape(1, 16, 16))
        y_predict = np.argmax(possibilities)
        str_possibilities = possibilities[0][y_predict]

        print("Ture:", i, "Guess:", y_predict)
        plt.figure(figsize=(7, 7))
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(linestyle='-.')
        a = img.reshape(128, 2)
        first_a = a[0:128, 0]
        last_a = a[0:128, 1]
        plt.scatter(first_a, last_a, cmap=plt.cm.jet,
                    c=radio11CamweightTestX.reshape(128), s=100)
        plt.colorbar();
        plt.title('True label:' + str(category[i]) + '   likelihood of label ' +
                  str(category[y_predict]) + ': ' + str(str_possibilities))
        outputfile_path = output_path + "/" + "label" + str(i) + str(category[i]) + "星座" + str(
            count) + ".png"
        count = count + 1
        plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')


    count = 0
    for img in adversarial_signal:  # 画对抗样本的识别结果类标的gradcam
        img = img.reshape(1, 16, 16)
        output_path = a_s_path
        cam, cam_weight = grad_cam(model, img, i, 'conv2d_9')
        radio11CamweightTestX = cam_weight
        possibilities = model.predict(img.reshape(1, 16, 16))
        y_predict = np.argmax(possibilities)
        str_possibilities = possibilities[0][y_predict]

        print("Ture:", i, "Guess:", y_predict)
        plt.figure(figsize=(7, 7))
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(linestyle='-.')
        a = img.reshape(128, 2)
        first_a = a[0:128, 0]
        last_a = a[0:128, 1]
        plt.scatter(first_a, last_a, cmap=plt.cm.jet,
                    c=radio11CamweightTestX.reshape(128), s=100)
        plt.colorbar();
        plt.title('True label:' + str(category[i]) + '   likelihood of label ' +
                  str(category[y_predict]) + ': ' + str(str_possibilities))
        outputfile_path = output_path + "/" + "label" + str(i) + str(category[i]) + "星座" + str(
            count) + ".png"
        count = count + 1
        plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')


    count = 0
    for img in source_signal:
        img = img.reshape(1, 16, 16)
        output_path = s_s_path
        count = count + 1
        cam, cam_weight = grad_cam(model, img, i, 'conv2d_9')
        radio11CamweightTestX = cam_weight
        possibilities = model.predict(img.reshape(1, 16, 16))
        y_predict = np.argmax(possibilities)
        str_possibilities = possibilities[0][y_predict]
        print("Ture:", i, "Guess:", y_predict)
        plt.figure(figsize=(7, 7))
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(linestyle='-.')
        a = img.reshape(128, 2)
        first_a = a[0:128, 0]
        last_a = a[0:128, 1]
        plt.scatter(first_a, last_a, cmap=plt.cm.jet,
                    c=radio11CamweightTestX.reshape(128), s=100)
        plt.colorbar();
        plt.title('True label:' + str(category[i]) + '   likelihood of label ' +
                  str(category[y_predict]) + ': ' + str(str_possibilities))
        outputfile_path = output_path + "/" + "label" + str(i) + str(category[i]) + "星座" + str(
            count) + ".png"
        count = count + 1
        plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')

