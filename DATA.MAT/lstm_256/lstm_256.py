from __future__ import division, print_function, absolute_import
# -*- coding: utf-8 -*-
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import tensorflow as tf
import h5py
from tflearn.datasets import imdb
import random
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#%%数据预处理
DATA_PATH = '/public/home/ch/files/'
#DATA_PATH = 'D:/F/radio/radio12_npy/'
trainX = np.load(DATA_PATH + 'radio12CNormTrainX.npy')
trainSnrY = np.load(DATA_PATH + 'radio12CNormTrainSnrY.npy')
trainSnr = trainSnrY[0]
trainY = trainSnrY[1] #SnrY中[0]是信噪比，[1]是trainY,即调制类型(0~11)
testX = np.load(DATA_PATH + 'radio12CNormTestX.npy')
testSnrY = np.load(DATA_PATH + 'radio12CNormTestSnrY.npy')
testSnr = testSnrY[0]
testY = testSnrY[1]

list_val = np.arange(len(testX)) #list_val==array([0,1,2,,...,155999])   len(testX)==156000
random.shuffle(list_val) #将序列的所有元素随机排序
val_percent = 0.1
valX = testX[list_val[0: int(len(testX) * val_percent)]]
valY = testY[list_val[0: int(len(testX) * val_percent)]]
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

num_classes = 12 #信号分类数

trainY = tflearn.data_utils.to_categorical(trainY,num_classes)
valY = tflearn.data_utils.to_categorical(valY,num_classes)
testY = tflearn.data_utils.to_categorical(testY,num_classes)
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

in_shp = list(trainX.shape[1:])
print (trainX.shape, in_shp)
#%%
batch_size = 100
# Network building
start = time.time()
input_layer = tflearn.input_data([None, 512, 2])

net_lstm1 = tflearn.lstm(input_layer, n_units=256, dropout=0.8, dynamic=True, return_seq=True)

net_lstm1 = tf.transpose(net_lstm1, [1,0,2])

print(net_lstm1[0].get_shape())

net_output = tflearn.fully_connected(net_lstm1, 12, activation='softmax')
net = tflearn.regression(net_output, optimizer='adam', learning_rate=0.0001,
                         loss='categorical_crossentropy')

# Training

model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="./logs", max_checkpoints=3, best_checkpoint_path="lstm_256units")
#model.load('model.tfl')

model.fit(trainX, trainY, validation_set=(valX, valY), show_metric=True,batch_size=batch_size, n_epoch=50, shuffle=True,run_id='lstm_256units')

end = time.time()
print (end-start)

model.save('model.tfl')


eva = model.evaluate(X=trainX, Y=trainY, batch_size = batch_size)
print(eva)

eva = model.evaluate(X=valX, Y=valY, batch_size = batch_size)
print(eva)

eva = model.evaluate(X=testX, Y=testY, batch_size = batch_size)
print(eva)

## confusion matrix
y_test = testSnrY[1]
y_predict = np.zeros(shape=len(testX), dtype=np.int32)
pre_batch = 1000
iter = int(len(testX)/pre_batch)
for i in range(0, iter):
    y_predict[i*pre_batch: (i+1)*pre_batch] = np.argmax(model.predict(testX[i*pre_batch: (i+1)*pre_batch]), axis=1)

import sklearn
test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)

np.save('TESTconfusion', (y_predict, y_test, testSnrY[0]))
np.save('CM',test_CM)
