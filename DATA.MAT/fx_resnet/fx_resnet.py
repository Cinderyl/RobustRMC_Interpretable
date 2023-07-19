import h5py
import numpy as np
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout
from keras import layers
import keras.models as Model
from keras.regularizers import *
from keras.optimizers import adam
import seaborn as sns
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import tflearn
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
data_format = 'channels_first'

def residual_stack(Xm,kennel_size,Seq,pool_size):
    #1*1 Conv Linear
    Xm = Conv2D(32, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    #Residual Unit 1
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv2", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #Residual Unit 2
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv4", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv5", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm
#%%
#input layer
Xm_input = Input(in_shp)
Xm = Reshape([1,in_shp[0],2], input_shape=in_shp)(Xm_input)
#Residual Srack
Xm = residual_stack(Xm,kennel_size=(3,2),Seq="ReStk0",pool_size=(2,2))
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk1",pool_size=(2,1))   
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk2",pool_size=(2,1))   
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk3",pool_size=(2,1))   
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk4",pool_size=(2,1))   
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk5",pool_size=(2,1))   

Xm = Flatten(data_format=data_format)(Xm)
Xm = Dense(128, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.3)(Xm)
#Full Con 2
Xm = Dense(num_classes, kernel_initializer='glorot_normal', name="dense2")(Xm)
#SoftMax
Xm = Activation('softmax')(Xm)
#Create Model
model = Model.Model(inputs=Xm_input,outputs=Xm)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

#%%Set up some params 
epochs = 100     # number of epochs to train on
batch_size = 64  # training batch size default 1024

#%%
filepath = 'resnet.h5' #所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
checkpath="weights.best.hdf5"
log_filename = 'resnet_train.csv' 
try:
    model.load_weights(filepath)
    print("加载模型成功!继续训练模型")
except :    
    print("加载模型失败!开始训练一个新模型")

history = model.fit(trainX,
    trainY,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valX, valY),
    callbacks = [ #回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
        keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
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
np.save('CM', (test_CM))