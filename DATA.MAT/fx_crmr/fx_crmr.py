# -*- coding: utf-8 -*-
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle,sys, keras
import h5py
import tflearn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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


trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[2],trainX.shape[1]))
valX = np.reshape(valX, (valX.shape[0],valX.shape[2],valX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],testX.shape[2],testX.shape[1]))
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

in_shp = list(trainX.shape[1:])
print (trainX.shape, in_shp)
#%%CNN模型
dr = 0.5 # dropout rate (%) 卷积层部分  https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d
model = models.Sequential() #这里使用keras的序贯模型  https://keras-cn.readthedocs.io/en/latest/models/sequential/
model.add(Reshape(([1]+in_shp), input_shape=in_shp))
model.add(ZeroPadding2D((0, 1),data_format="channels_first")) #Keras config file at `keras.json`.If you never set it, then it will be "channels_last".
model.add(Conv2D(256, (1, 3),padding='valid', activation="relu",
                 name="conv1", init='glorot_uniform',data_format="channels_first")) #默认strides=1
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 1),data_format="channels_first"))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu",
                 name="conv2", init='glorot_uniform',data_format="channels_first"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( num_classes, init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([num_classes]))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary() #layer.input_shape用于获取layer输入形状
#%%Set up some params 
epochs = 100     # number of epochs to train on
batch_size = 64  # training batch size default1024
#%%
filepath = 'convmodrecnets_CNN2_0.5.wts.h5' #所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
checkpath="weights.best.hdf5"
log_filename = 'model_train.csv' 
try:
    model.load_weights("convmodrecnets_CNN2_0.5.wts.h5")
    print("加载模型成功!继续训练模型")
except :    
    print("加载模型失败!开始训练一个新模型")
    
    
history = model.fit(trainX,
    trainY,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(valX, valY),
    callbacks = [ #回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
        keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, 
                                    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
        keras.callbacks.ModelCheckpoint(checkpath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
       # keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
    ]) #EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
    
    
model_name = checkpath
model.save(checkpath)
model.save_weights(checkpath)

try:
    model.load_weights(checkpath)
    print("加载模型成功!")
except :    
    print("加载模型失败!")
#%%
score = model.evaluate(trainX, trainY, verbose=0, batch_size=batch_size)
print(score)    
    
score = model.evaluate(valX, valY, verbose=0, batch_size=batch_size)
print(score)  

score = model.evaluate(testX, testY, verbose=0, batch_size=batch_size)
print(score)
#%%
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
#%%
'''
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%
# Plot confusion matrix 画图
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
#    print(test_SNRs)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
#%%
# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
'''