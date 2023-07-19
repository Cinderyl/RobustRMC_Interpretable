import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import tensorflow as tf
import h5py
from tflearn.datasets import imdb
import random
import os, time
import keras
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Input
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,MaxPool2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape,Dropout
from keras.layers import Lambda
from keras import regularizers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#%%数据预处理
#DATA_PATH = 'D:/F/radio/DeepSig/RML2016.10a.pkl/'
DATA_PATH = '/data0/quziwen/yelinhui/explainable/dataset/'
trainX = np.load(DATA_PATH + 'radio11CNormTrainX.npy')
trainSnrY = np.load(DATA_PATH + 'radio11CNormTrainSnrY.npy')
trainSnr = trainSnrY[0]
trainY = trainSnrY[1] #SnrY中[0]是信噪比，[1]是trainY,即调制类型(0~11)
testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')
testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy')
testSnr = testSnrY[0]
testY = testSnrY[1]

list_val = np.arange(len(testX)) #list_val==array([0,1,2,,...,155999])   len(testX)==156000
random.shuffle(list_val) #将序列的所有元素随机排序
val_percent = 0.1
valX = testX[list_val[0: int(len(testX) * val_percent)]]
valY = testY[list_val[0: int(len(testX) * val_percent)]]
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))
# (176000, 128, 2) (176000,) (4400, 128, 2) (4400,) (44000, 128, 2) (44000,)

num_classes = 11 #信号分类数
trainY = tflearn.data_utils.to_categorical(trainY,num_classes)   # 将索引转换为one-hot编码
valY = tflearn.data_utils.to_categorical(valY,num_classes)
testY = tflearn.data_utils.to_categorical(testY,num_classes)
print(np.shape(trainX), np.shape(trainY),np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))

weight_decay = 0.0001
dropout = 0.5

in_shp = list(trainX.shape[1:])
print(in_shp)

weight_decay = 0.0001  # 新增


def lstm(X):
    # X = Input(shape=(128, 2,))
    x = LSTM(units=256, return_sequences=True)(X)
    # x = Reshape((128, 256, 1), input_shape=(128, 256))(x)
    # x = Flatten()(x)
    # x = Dense(11, activation='softmax')(x)
    # model = Model(X, x)
    return x

def cnn(X):
    # X = Input(shape=(16, 16, 1,))
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(X)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D(pool_size=(2, 2), strides=(1,1),padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same')(x)

    # x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)
    #
    # x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Reshape((128, 256))(x)
    # x = Flatten()(x)
    # x = Dense(256, kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    # x = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    # x = Dropout(0.5)(x)
    # x = Dense(11)(x)
    # x = Activation('softmax')(x)
    # model = Model(X, x)
    return x

# cnn().summary()
# lstm().summary()

def concate(x1):
    return K.concatenate([x1[0], x1[1]], axis=1)

def sum(x):
    return x[0]+x[1]

def model_e():
    input_x = Input(shape=(128, 2,))
    x_cnn = Reshape((16, 16, 1))(input_x)
    x_cnn = cnn(x_cnn)            # [16,16,128]
    # x_cnn = Reshape((128, 256))(x_cnn)
    x_lstm = lstm(input_x)

    x_output = [x_cnn, x_lstm]
    x_output = Lambda(concate)(x_output)
    x_output = Flatten()(x_output)
    x_output = Dense(256, activation='relu')(x_output)
    x_output = Dense(128, activation='relu')(x_output)
    x_output = Dense(11, activation='softmax')(x_output)
    model = Model(inputs=input_x, output=x_output)
    return model

model = model_e()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50  # number of epochs to train on
batch_size = 32  # training batch size default 1024
filepath = './e_model_vgg16_lstm_cancate_0.55925_区分较明显.h5'  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
try:
    model.load_weights(filepath)
    print("加载模型成功!")
except :
    print("加载模型失败!")
log_filename = 'model_train_vgg16_lstm_cancate_1.csv'
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
# model_name = filepath
# model.save(filepath)
# model.save_weights(filepath)
# try:
#     model.load_weights(filepath)
#     print("加载模型成功!")
# except :
#     print("加载模型失败!")

'''
测试模型性能
'''
c = 0
snr = 18
index = []
a = np.array([snr], dtype=np.float64)    # 需要测试的信号的信噪比和索引
count = 0
for i in testSnrY.T:                         # 找所需要测试的信号的索引，count其实就是找到的索引，0是snr，1是信号标签
    if i[0] == a[0]:
        index.append(count)
    count = count+1
index8 = np.array(index)    # 所找到的索引
train = testX[index]
testX = train
y_test = testSnrY[1]
y_test = y_test[index]

y_predict = np.zeros(shape=len(testX), dtype=np.int32)    # 44000
pre_batch = 1000
iter = int(len(testX) / pre_batch)   # 44
for i in range(0, iter):
    y_predict[i * pre_batch: (i + 1) * pre_batch] = np.argmax(model.predict(testX[i * pre_batch: (i + 1) * pre_batch]),
                                                              axis=1)
import sklearn
test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)
np.save("CM", test_CM)


'''
测试模型在各个信噪比下对各个类的识别准确率
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
# for c in range(11):
#     colum = colum + 1
#     worksheet.write(0, colum, str(category[c]))
#     row = 0
#
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
#         train = np.reshape(train, (len(train), 128, 2));   # 测试数据
#         result = model.predict(train)
#         result = np.argmax(result, axis=1)
#
#         import sklearn
#         y_test = testSnrY[1]
#         y_test = y_test[index]
#         test_acc = sklearn.metrics.accuracy_score(y_test, result)
#         worksheet.write(row, colum, test_acc)
#         result_save[row-1, colum-1] = test_acc
#         np.save("./集成模型对各个信噪比下各个信号的准确率_vgg16_lstm_cancate_1.npy", result_save)
#         print("对应类别的类标{}_信噪比{}_准确率{}".format(c, snr, test_acc))
#     workbook.save('./模型的识别准确率_vgg16_lstm_cancate_1.xls')
