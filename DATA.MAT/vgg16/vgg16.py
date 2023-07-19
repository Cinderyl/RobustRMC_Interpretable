# -*- coding: utf-8 -*-
from keras.models import Sequential  
from keras.layers import Input, Dense, Dropout, Activation,Reshape
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers
import keras
import numpy as np
import os,random
import tflearn
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Building 'VGG Network'
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
valX = testX[list_val[0: int(len(testX) * val_percent)]]     # 验证集，占训练集的十分之一
valY = testY[list_val[0: int(len(testX) * val_percent)]]     # 测试集，占训练集的十分之一
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

num_classes = 12 #信号分类数

trainY = tflearn.data_utils.to_categorical(trainY,num_classes)
valY = tflearn.data_utils.to_categorical(valY,num_classes)
testY = tflearn.data_utils.to_categorical(testY,num_classes)
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

trainX = np.reshape(trainX,(len(trainX),32,32));
valX = np.reshape(valX,(len(valX),32,32));
testX = np.reshape(testX,(len(testX),32,32));
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

in_shp = list(trainX.shape[1:])
print (trainX.shape, in_shp)
#%%
#用于正则化时权重降低的速度
weight_decay = 0.0005
nb_epoch=50
batch_size=32
#layer1 32*32*3
start = time.time()
model = Sequential()
#第一个卷积层的卷积核的数目是64 ，卷积核的大小是3*3，stride没写，默认应该是1*1
#对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
model.add(Reshape((in_shp+[1]), input_shape=in_shp))
model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
#进行一次归一化
model.add(BatchNormalization())
model.add(Dropout(0.3))
#layer2 32*32*64
model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
#padding默认是same，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')  )
#layer3 16*16*64
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer4 16*16*128
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer5 8*8*128
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer6 8*8*256
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer7 8*8*256
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer8 4*4*256
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer9 4*4*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer10 4*4*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer11 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer12 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer13 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#layer14 1*1*512
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#layer15 512
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#layer16 512
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
# 11
model.summary()     # 输出模型结构信息
#%%
filepath = 'vgg16.h5' #所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
log_filename = 'model_vgg16.csv' 
try:
    model.load_weights(filepath)
    print("加载模型成功!继续训练模型")
except :    
    print("加载模型失败!开始训练一个新模型")
    
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(trainX,trainY,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(valX, valY),
    callbacks = [ #回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
        keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, 
                                    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
       # keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
    ]) #EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch

end = time.time()
print (end-start)
model_name = filepath
model.save(filepath)
model.save_weights(filepath)

try:
    model.load_weights(filepath)
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

np.save('/public/home/ch/MSH/model_VGG/TESTconfusion', (y_predict, y_test, testSnrY[0]))
np.save('CM',test_CM)