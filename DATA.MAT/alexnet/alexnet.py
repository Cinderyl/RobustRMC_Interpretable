from keras.models import Sequential  
from keras.layers import Input, Dense, Dropout, Activation,Reshape
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten
import keras
import numpy as np
import os,random
import tflearn
 
 
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#%%数据预处理
#DATA_PATH = '/public/home/ch/files/'
DATA_PATH = 'D:/F/radio/radio12_npy/'
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

trainX = np.reshape(trainX,(len(trainX),32,32));
valX = np.reshape(valX,(len(valX),32,32));
testX = np.reshape(testX,(len(testX),32,32));
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY),np.shape(testX), np.shape(testY))

in_shp = list(trainX.shape[1:])
print (trainX.shape, in_shp)
#%%
data_format = 'channels_last'
nn = Sequential() 

nn.add(Reshape((in_shp+[1]), input_shape=in_shp))

nn.add(Conv2D(32,(3,3), activation='relu',data_format=data_format))
nn.add(MaxPooling2D(pool_size=(2, 2),strides=1,data_format=data_format))
 
nn.add(Conv2D(64,(3,3), activation='relu',data_format=data_format))
nn.add(MaxPooling2D(pool_size=(2, 2),data_format=data_format))
 
nn.add(Conv2D(128,(3,3), activation='relu',data_format=data_format))
nn.add(MaxPooling2D(pool_size=(2, 2),data_format=data_format))
 
nn.add(Conv2D(256,(3,3), activation='relu',data_format=data_format))
nn.add(MaxPooling2D(pool_size=(2, 2),data_format=data_format))
 
nn.add(Flatten())
 
nn.add(Dense(500)) 
nn.add(Activation('relu')) 
nn.add(Dropout(0.5)) 
 
'''
nn.add(Dense(500)) 
nn.add(Activation('relu'))
nn.add(Dropout(0.5))
 
nn.add(Dense(500)) 
nn.add(Activation('relu'))
'''
nn.add(Dense(num_classes))
nn.add(Activation('softmax'))
 
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.summary()   

#%%Set up some params 
epochs = 100     # number of epochs to train on
batch_size = 64  # training batch size default 1024

#%%
filepath = 'alexnet.h5' #所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
log_filename = 'model_train.csv' 
try:
    nn.load_weights(filepath)
    print("加载模型成功!继续训练模型")
except :    
    print("加载模型失败!开始训练一个新模型")
   
history = nn.fit(trainX,
    trainY,
    batch_size=batch_size,
    epochs=epochs,
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
   
    
model_name = filepath
nn.save(filepath)
nn.save_weights(filepath)

try:
    nn.load_weights(filepath)
    print("加载模型成功!")
except :    
    print("加载模型失败!")
#%%
score = nn.evaluate(trainX, trainY, verbose=0, batch_size=batch_size)
print(score)    
    
score = nn.evaluate(valX, valY, verbose=0, batch_size=batch_size)
print(score)  

score = nn.evaluate(testX, testY, verbose=0, batch_size=batch_size)
print(score)
#%%
## confusion matrix
y_test = testSnrY[1]
y_predict = np.zeros(shape=len(testX), dtype=np.int32)
pre_batch = 1000
iter = int(len(testX)/pre_batch)
for i in range(0, iter):
    y_predict[i*pre_batch: (i+1)*pre_batch] = np.argmax(nn.predict(testX[i*pre_batch: (i+1)*pre_batch]), axis=1)

import sklearn
test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)

np.save('TESTconfusion', (y_predict, y_test, testSnrY[0]))
np.save('CM',test_CM)