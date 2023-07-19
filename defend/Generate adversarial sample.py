import xlwt
from keras.layers.core import Lambda
import matplotlib.pyplot as plt
plt.ion()
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
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, MaxPool2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape, Dropout
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import foolbox

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
category = {0:"WBFM", 1:"QPSK", 2:"QAM64", 3: "QAM16", 4:"PAM4", 5:"GFSK", 6:"CPFSK" ,7:"BPSK", 8:"8PSK", 9:"AM-SSB", 10:"AM-DSB"}

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

trainX = np.reshape(trainX, (len(trainX), 16, 16))
valX = np.reshape(valX, (len(valX), 16, 16))
testX = np.reshape(testX, (len(testX), 16, 16))
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))


in_shp = list(trainX.shape[1:])
weight_decay = 0.0001
dropout = 0.5

def build_model():
    model = Sequential()
    model.add(Reshape((in_shp + [1]), input_shape=in_shp))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(num_classes, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

model = build_model()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 100  # number of epochs to train on
batch_size = 64  # training batch size default 1024

filepath = 'NIN.h5'  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
try:
    model.load_weights(filepath)
    print("加载模型成功!")
except:
    print("加载模型失败!")


attack_name = ['CW', 'FGSM', 'LB-FGSM', 'JSMA', 'IBM', 'MI-FGSM', 'DeepFool',
               'PGD', 'DeepFoolL2',  # 0-8
               'DeepFoolLinf', 'AdditiveGaussian', 'SaltAndPepper', 'Boundary',
               'NewtonFool', 'RandomPGD']


'''
将第4类攻击为第7类，将第7类攻击为第4类
限制噪声扰动的情况下---FGSM
'''
# c = 4
# index = []
# a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
# y_test = testSnrY[1]
# y_test = y_test[index]                             # 找出的标签
# result = model.predict(train)
# result = np.argmax(result, axis=1)
#
# print("----选择目标类中能够被正确识别的样本----")
# count = 0
# right_index = []
# for k in result:
#     if k == a[1]:
#         right_index.append(count)
#     count = count + 1
# if right_index != []:
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#     right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
# else:
#     right_train = []
#     right_y = []
#
# # # 多个样本进行攻击
# count = 0         # 用来计算攻击成功的样本数
# count_index = []  # 用来存放攻击成功的样本的索引
# index = 0         # 用来打印哪一个样本攻击失败了
# ad_siganl_save = []
# fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
# attack = foolbox.attacks.FGSM(fmodel, TargetClass(7))
# for example in right_train:
#     index = index+1
#     print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#     # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#     ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1)   # 限制加上去的最大幅值为0.1
#
#     if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#         ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#         count = count+1                     # 计算攻击成功的样本的个数
#         count_index.append(index-1)
#     else:
#         print("第%d个样本攻击失败了" % index)
#         print("-------------------")
#     # if count == 20:
#     #     break
#
# ad_siganl_save = np.array(ad_siganl_save)
# count_index = np.array(count_index)
# source_signal = right_train[count_index]
#
# save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/4_to_7_限制幅值为0.1"
# if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#     pass
# else:                                   # 若不存在，则创建这个文件夹
#     os.makedirs(save_path)
# np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
# # 保存原信号
#
# # 计算对抗信号的信噪比
# save_snr = []
# for i in range(len(ad_siganl_save)):
#     example_ad = ad_siganl_save[i]
#     example_siganl = source_signal[i]
#
#     signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#     noise_of_signal = example_siganl - signal
#     snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#     # print(snr1)
#
#     noise_of_example_ad = example_ad - signal
#     snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#     save_snr.append(snr)
#
# result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
# result_ad = np.argmax(result_ad, axis=1)
# # 创建excel表格
# workbook = xlwt.Workbook(encoding='ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# style = xlwt.XFStyle()  # 初始化样式
# font = xlwt.Font()  # 为样式创建字体
# font.name = 'Times New Roman'
# font.bold = True  # 黑体
# font.underline = True  # 下划线
# font.italic = True  # 斜体字
# style.font = font  # 设定样式
# worksheet.write(0, 0, '对抗样本序号')
# worksheet.write(0, 1, '识别结果')
# worksheet.write(0, 2, '攻击成功率')
# worksheet.write(0, 3, '对抗样本的信噪比')
# worksheet.write(1, 2, count / len(right_train))
#
# c = 0
# for r in result_ad:
#     print(int(r))
#     worksheet.write(c + 1, 0, c)  # 写入序号
#     worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#     worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#     c = c + 1
#
# workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
# np.save(save_path+"/"+"不限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)


'''
将第7类攻击为第4类
限制噪声扰动的情况下---FGSM
'''
# c = 7
# index = []
# a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
# y_test = testSnrY[1]
# y_test = y_test[index]                             # 找出的标签
# result = model.predict(train)
# result = np.argmax(result, axis=1)
#
# print("----选择目标类中能够被正确识别的样本----")
# count = 0
# right_index = []
# for k in result:
#     if k == a[1]:
#         right_index.append(count)
#     count = count + 1
# if right_index != []:
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#     right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
# else:
#     right_train = []
#     right_y = []
#
# # # 多个样本进行攻击
# count = 0         # 用来计算攻击成功的样本数
# count_index = []  # 用来存放攻击成功的样本的索引
# index = 0         # 用来打印哪一个样本攻击失败了
# ad_siganl_save = []
# fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
# attack = foolbox.attacks.FGSM(fmodel, TargetClass(4))
# for example in right_train:
#     index = index+1
#     print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#     # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#     ad_siganl = attack(example, label=right_y[0], max_epsilon=0.15)   # 限制加上去的最大幅值为0.1
#
#     if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#         ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#         count = count+1                     # 计算攻击成功的样本的个数
#         count_index.append(index-1)
#     else:
#         print("第%d个样本攻击失败了" % index)
#         print("-------------------")
#     # if count == 20:
#     #     break
#
# ad_siganl_save = np.array(ad_siganl_save)
# count_index = np.array(count_index)
# source_signal = right_train[count_index]
#
# save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1"
# if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#     pass
# else:                                   # 若不存在，则创建这个文件夹
#     os.makedirs(save_path)
# np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
# # 保存原信号
#
# # 计算对抗信号的信噪比
# save_snr = []
# for i in range(len(ad_siganl_save)):
#     example_ad = ad_siganl_save[i]
#     example_siganl = source_signal[i]
#
#     signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#     noise_of_signal = example_siganl - signal
#     snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#     # print(snr1)
#
#     noise_of_example_ad = example_ad - signal
#     snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#     save_snr.append(snr)
#
# result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
# result_ad = np.argmax(result_ad, axis=1)
# # 创建excel表格
# workbook = xlwt.Workbook(encoding='ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# style = xlwt.XFStyle()  # 初始化样式
# font = xlwt.Font()  # 为样式创建字体
# font.name = 'Times New Roman'
# font.bold = True  # 黑体
# font.underline = True  # 下划线
# font.italic = True  # 斜体字
# style.font = font  # 设定样式
# worksheet.write(0, 0, '对抗样本序号')
# worksheet.write(0, 1, '识别结果')
# worksheet.write(0, 2, '攻击成功率')
# worksheet.write(0, 3, '对抗样本的信噪比')
# worksheet.write(1, 2, count / len(right_train))
#
# c = 0
# for r in result_ad:
#     print(int(r))
#     worksheet.write(c + 1, 0, c)  # 写入序号
#     worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#     worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#     c = c + 1
#
# workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
# np.save(save_path+"/"+"限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)


'''
限制扰动的无目标攻击---FGSM
'''
# for c in range(11):
#     index = []
#     a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)    # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
#     y_test = testSnrY[1]
#     y_test = y_test[index]                             # 找出的标签
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#
#     print("----选择目标类中能够被正确识别的样本----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#     if right_index != []:
#         right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#         right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#         right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
#     else:
#         right_train = []
#         right_y = []
#
#     # # 多个样本进行攻击
#     count = 0         # 用来计算攻击成功的样本数
#     count_index = []  # 用来存放攻击成功的样本的索引
#     index = 0         # 用来打印哪一个样本攻击失败了
#     ad_siganl_save = []
#     fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
#     attack = foolbox.attacks.FGSM(fmodel)
#     for example in right_train:
#         index = index+1
#         print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#         # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#         ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1)   # 限制加上去的最大幅值为0.1
#
#         if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#         else:
#             print("第%d个样本攻击失败了" % index)
#             print("-------------------")
#         if count == 20:
#             break
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#
#     save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/无目标攻击-限制噪声幅值0.1/"\
#                 +"label_{}{}".format(right_y[0], category[right_y[0]])
#     if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#         pass
#     else:                                   # 若不存在，则创建这个文件夹
#         os.makedirs(save_path)
#     np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
#     # 保存原信号
#
#     # 计算对抗信号的信噪比
#     save_snr = []
#     for i in range(len(ad_siganl_save)):
#         example_ad = ad_siganl_save[i]
#         example_siganl = source_signal[i]
#
#         signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#         noise_of_signal = example_siganl - signal
#         snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#         # print(snr1)
#
#         noise_of_example_ad = example_ad - signal
#         snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#         save_snr.append(snr)
#
#     result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
#     result_ad = np.argmax(result_ad, axis=1)
#     # 创建excel表格
#     workbook = xlwt.Workbook(encoding='ascii')
#     worksheet = workbook.add_sheet('My Worksheet')
#     style = xlwt.XFStyle()  # 初始化样式
#     font = xlwt.Font()  # 为样式创建字体
#     font.name = 'Times New Roman'
#     font.bold = True  # 黑体
#     font.underline = True  # 下划线
#     font.italic = True  # 斜体字
#     style.font = font  # 设定样式
#     worksheet.write(0, 0, '对抗样本序号')
#     worksheet.write(0, 1, '识别结果')
#     worksheet.write(0, 2, '攻击成功率')
#     worksheet.write(0, 3, '对抗样本的信噪比')
#     worksheet.write(1, 2, count / len(right_train))
#
#     c = 0
#     for r in result_ad:
#         print(int(r))
#         worksheet.write(c + 1, 0, c)  # 写入序号
#         worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#         worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#         c = c + 1
#
#     workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
#     np.save(save_path+"/"+"不限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)

'''
无目标攻击，限制噪声的扰动，用于生成对抗样本
考虑类标以及信噪比
'''
# 找目标类的数据，也就是生成某一类信号的对抗样本
total_ad = 5000       # 需要生成的对抗样本总量
train_total_ad = 4000    # 用来放到训练集中的对抗样本
test_total_ad = 1000       # 拿来测试的对抗样本

for c in range(11):
    index = []
    a = np.array([18, c], dtype=np.float64)       # 需要测试的信号的信噪比和索引
    count = 0
    for i in trainSnrY.T:                         # 找所需要测试的信号的索引
        if i[1] == a[1]:
            index.append(count)
        count = count+1
    index = np.array(index)    # 找到目标类的索引
    train = trainX[index]       # 目标类的数据
    train = np.reshape(train, (len(train), 16, 16))   # 找出的需要攻击的信号,[16000, 128,2]
    y_test = trainSnrY[1]
    y_test = y_test[index]      # 目标类的类标
    result = model.predict(train)
    result = np.argmax(result, axis=1)
    print(train.shape)

    print("----processing data----")
    print("----选择测试集中预测正确的数据进行攻击----")
    count = 0
    right_index = []
    for k in result:
        if k == a[1]:
            right_index.append(count)
        count = count + 1

    if right_index != []:
        right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
        right_train = train[right_index]  # 找到的所有预测正确的数据
        right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标

    else:
        right_train = []
        right_y = []  # 找到的所有预测正确的数据的类标


    #多个样本进行攻击
    count = 0         # 用来计算攻击成功的样本数
    count_index = []  # 用来存放攻击成功的样本的索引
    index = 0         # 用来打印哪一个样本攻击失败了
    ad_siganl_save = []
    fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
    attack = foolbox.attacks.FGSM(fmodel)
    for example in right_train:
        index = index+1
        print("开始处理第{}类的第{}个样本".format(c, index))
        # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
        ad_siganl = attack(example, label=c, max_epsilon=0.3)   # 限制加上去的最大幅值为0.1

        if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
            ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
            count = count+1                     # 计算攻击成功的样本的个数
            count_index.append(index-1)
        else:
            print("第%d个样本攻击失败了" % index)
            print("-------------------")
        if count == total_ad:
            break

    ad_siganl_save = np.array(ad_siganl_save)
    count_index = np.array(count_index)
    source_signal = right_train[count_index]


    trainX = np.concatenate((trainX, ad_siganl_save[0:train_total_ad]), axis=0)   # 用于对抗训练的训练集
    print(trainX.shape)

    save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/"\
                +"label_{}{}".format(c, category[c])
    if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
        pass
    else:                                   # 若不存在，则创建这个文件夹
        os.makedirs(save_path)

    # 保存原信号

    # 计算对抗信号的信噪比
    save_snr = []
    ad_snrY = []
    for i in range(len(ad_siganl_save)):
        example_ad = ad_siganl_save[i]
        example_siganl = source_signal[i]

        signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
        noise_of_signal = example_siganl - signal
        snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))

        noise_of_example_ad = example_ad - signal
        snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
        snr = int(snr)
        snr = int((snr+1)/2)*2
        example_snrY = [snr, c]
        ad_snrY.append(example_snrY)          # 保存对抗样本的信噪比和类标
        save_snr.append(snr)
    ad_snrY = np.array(ad_snrY).T               # 保存对抗样本的信噪比和类标

    trainSnrY = np.concatenate((trainSnrY, ad_snrY[0:train_total_ad]), axis=1)   # 含有对抗样本的类标
    print('----------')
    print(trainSnrY.shape)

    result_ad = model.predict(ad_siganl_save.reshape(-1, 16, 16))   # 对抗信号识别结果
    result_ad = np.argmax(result_ad, axis=1)
    # 创建excel表格
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('My Worksheet')
    style = xlwt.XFStyle()  # 初始化样式
    font = xlwt.Font()  # 为样式创建字体
    font.name = 'Times New Roman'
    font.bold = True  # 黑体
    font.underline = True  # 下划线
    font.italic = True  # 斜体字
    style.font = font  # 设定样式
    worksheet.write(0, 0, '对抗样本序号')
    worksheet.write(0, 1, '识别结果')
    worksheet.write(0, 2, '攻击成功率')
    worksheet.write(0, 3, '对抗样本的信噪比')
    worksheet.write(1, 2, count / len(train))

    c1 = 0
    for r in result_ad:
        print(int(r))
        worksheet.write(c1 + 1, 0, c1)  # 写入序号
        worksheet.write(c1 + 1, 3, int(save_snr[c1]))    # 写入对抗样本的信噪比
        worksheet.write(c1 + 1, 1, int(r))  # 写入对抗样本的识别结果
        c1 = c1 + 1

    workbook.save(save_path+"/"+'原类标为{}.xls'.format(c))  # 保存文件

    path_train = save_path + "/" + "每类用于训练的{}个对抗样本/".format(train_total_ad)
    if os.path.exists(path_train):  # 判断存储原信号的文件是否存在
        pass
    else:  # 若不存在，则创建这个文件夹
        os.makedirs(path_train)
    path_test = save_path+"/"+"每类用于测试的{}个对抗样本/".format(test_total_ad)
    if os.path.exists(path_test):  # 判断存储原信号的文件是否存在
        pass
    else:  # 若不存在，则创建这个文件夹
        os.makedirs(path_test)

    np.save(path_train + "限制噪声幅值label_{}_source_siganl_save.npy".format(c), source_signal[0:train_total_ad])
    np.save(path_test + "限制噪声幅值label_{}_source_sognal.npy".format(c), source_signal[-test_total_ad:])

    np.save(path_train+"对抗样本信噪比和真实类标.npy", ad_snrY[:, 0:train_total_ad])
    np.save(path_test+"对抗样本信噪比和真实类标.npy", ad_snrY[:, -test_total_ad:])

    np.save(path_train + "限制噪声幅值label_{}_ad_siganl_save.npy".format(c), ad_siganl_save[0:train_total_ad])
    np.save(path_test + "限制噪声幅值label_{}_ad_siganl_save.npy".format(c), ad_siganl_save[-test_total_ad:])

save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/NIN/对抗训练/delate/"
if os.path.exists(save_path):  # 判断存储原信号的文件是否存在
    pass
else:  # 若不存在，则创建这个文件夹
    os.makedirs(save_path)
np.save(save_path + "/trainX_ad.npy", trainX)
np.save(save_path + "/trainSnrY_ad.npy", trainSnrY)


'''
限制扰动的无目标攻击---MI-FGSM
'''
# for c in range(11):
#     index = []
#     a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)    # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
#     y_test = testSnrY[1]
#     y_test = y_test[index]                             # 找出的标签
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#
#     print("----选择目标类中能够被正确识别的样本----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#     if right_index != []:
#         right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#         right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#         right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
#     else:
#         right_train = []
#         right_y = []
#
#     # # 多个样本进行攻击
#     count = 0         # 用来计算攻击成功的样本数
#     count_index = []  # 用来存放攻击成功的样本的索引
#     index = 0         # 用来打印哪一个样本攻击失败了
#     ad_siganl_save = []
#     fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
#     attack = foolbox.attacks.IterativeGradientSignAttack(fmodel)
#     for example in right_train:
#         index = index+1
#         print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#         # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#         ad_siganl = attack(example, label=right_y[0])   # 限制加上去的最大幅值为0.1
#
#         if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#         else:
#             print("第%d个样本攻击失败了" % index)
#             print("-------------------")
#         if count == 20:
#             break
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#
#     save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/Experimental Result/其他攻击方法/LSTM/MI-FGSM/"\
#                 +"label_{}{}".format(right_y[0], category[right_y[0]])
#     if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#         pass
#     else:                                   # 若不存在，则创建这个文件夹
#         os.makedirs(save_path)
#     np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
#     # 保存原信号
#
#     # 计算对抗信号的信噪比
#     save_snr = []
#     for i in range(len(ad_siganl_save)):
#         example_ad = ad_siganl_save[i]
#         example_siganl = source_signal[i]
#
#         signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#         noise_of_signal = example_siganl - signal
#         snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#         # print(snr1)
#
#         noise_of_example_ad = example_ad - signal
#         snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#         save_snr.append(snr)
#
#     result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
#     result_ad = np.argmax(result_ad, axis=1)
#     # 创建excel表格
#     workbook = xlwt.Workbook(encoding='ascii')
#     worksheet = workbook.add_sheet('My Worksheet')
#     style = xlwt.XFStyle()  # 初始化样式
#     font = xlwt.Font()  # 为样式创建字体
#     font.name = 'Times New Roman'
#     font.bold = True  # 黑体
#     font.underline = True  # 下划线
#     font.italic = True  # 斜体字
#     style.font = font  # 设定样式
#     worksheet.write(0, 0, '对抗样本序号')
#     worksheet.write(0, 1, '识别结果')
#     worksheet.write(0, 2, '攻击成功率')
#     worksheet.write(0, 3, '对抗样本的信噪比')
#     worksheet.write(1, 2, count / len(right_train))
#
#     c = 0
#     for r in result_ad:
#         print(int(r))
#         worksheet.write(c + 1, 0, c)  # 写入序号
#         worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#         worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#         c = c + 1
#
#     workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
#     np.save(save_path+"/"+"不限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)


'''
将第4类攻击为第7类，将第7类攻击为第4类
限制噪声扰动的情况下---MI-FGSM
'''
# c = 4
# index = []
# a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
# y_test = testSnrY[1]
# y_test = y_test[index]                             # 找出的标签
# result = model.predict(train)
# result = np.argmax(result, axis=1)
#
# print("----选择目标类中能够被正确识别的样本----")
# count = 0
# right_index = []
# for k in result:
#     if k == a[1]:
#         right_index.append(count)
#     count = count + 1
# if right_index != []:
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#     right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
# else:
#     right_train = []
#     right_y = []
#
# # # 多个样本进行攻击
# count = 0         # 用来计算攻击成功的样本数
# count_index = []  # 用来存放攻击成功的样本的索引
# index = 0         # 用来打印哪一个样本攻击失败了
# ad_siganl_save = []
# fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
# attack = foolbox.attacks.IterativeGradientSignAttack(fmodel, TargetClass(7))
# for example in right_train:
#     index = index+1
#     print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#     # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#     ad_siganl = attack(example, label=right_y[0])   # 限制加上去的最大幅值为0.1
#
#     if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#         ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#         count = count+1                     # 计算攻击成功的样本的个数
#         count_index.append(index-1)
#     else:
#         print("第%d个样本攻击失败了" % index)
#         print("-------------------")
#     if count == 20:
#         break
#
# ad_siganl_save = np.array(ad_siganl_save)
# count_index = np.array(count_index)
# source_signal = right_train[count_index]
#
# save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/Experimental Result/其他攻击方法/LSTM/MI-FGSM_4_to_7"
# if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#     pass
# else:                                   # 若不存在，则创建这个文件夹
#     os.makedirs(save_path)
# np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
# # 保存原信号
#
# # 计算对抗信号的信噪比
# save_snr = []
# for i in range(len(ad_siganl_save)):
#     example_ad = ad_siganl_save[i]
#     example_siganl = source_signal[i]
#
#     signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#     noise_of_signal = example_siganl - signal
#     snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#     # print(snr1)
#
#     noise_of_example_ad = example_ad - signal
#     snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#     save_snr.append(snr)
#
# result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
# result_ad = np.argmax(result_ad, axis=1)
# # 创建excel表格
# workbook = xlwt.Workbook(encoding='ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# style = xlwt.XFStyle()  # 初始化样式
# font = xlwt.Font()  # 为样式创建字体
# font.name = 'Times New Roman'
# font.bold = True  # 黑体
# font.underline = True  # 下划线
# font.italic = True  # 斜体字
# style.font = font  # 设定样式
# worksheet.write(0, 0, '对抗样本序号')
# worksheet.write(0, 1, '识别结果')
# worksheet.write(0, 2, '攻击成功率')
# worksheet.write(0, 3, '对抗样本的信噪比')
# worksheet.write(1, 2, count / len(right_train))
#
# c = 0
# for r in result_ad:
#     print(int(r))
#     worksheet.write(c + 1, 0, c)  # 写入序号
#     worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#     worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#     c = c + 1
#
# workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
# np.save(save_path+"/"+"限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)


'''
将第7类攻击为第4类
限制噪声扰动的情况下---MI-FGSM
'''
# c = 7
# index = []
# a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# train = np.reshape(train, (len(train), 128, 2));   # 找出的需要攻击的信号
# y_test = testSnrY[1]
# y_test = y_test[index]                             # 找出的标签
# result = model.predict(train)
# result = np.argmax(result, axis=1)
#
# print("----选择目标类中能够被正确识别的样本----")
# count = 0
# right_index = []
# for k in result:
#     if k == a[1]:
#         right_index.append(count)
#     count = count + 1
# if right_index != []:
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]  # 找到的所有预测正确的数据,这里只取1个画图
#     right_y = y_test[right_index]  # 找到的所有预测正确的数据的类标
# else:
#     right_train = []
#     right_y = []
#
# # # 多个样本进行攻击
# count = 0         # 用来计算攻击成功的样本数
# count_index = []  # 用来存放攻击成功的样本的索引
# index = 0         # 用来打印哪一个样本攻击失败了
# ad_siganl_save = []
# fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
# attack = foolbox.attacks.IterativeGradientSignAttack(fmodel, TargetClass(4))
# for example in right_train:
#     index = index+1
#     print("开始处理第{}类的第{}个样本".format(right_y[0], index))
#     # attack = foolbox.attacks.FGSM(fmodel)                          # 无目标攻击
#     ad_siganl = attack(example, label=right_y[0])   # 限制加上去的最大幅值为0.1
#
#     if np.sum(ad_siganl) != None:           # 这里也就是判断攻击是否成功
#         ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#         count = count+1                     # 计算攻击成功的样本的个数
#         count_index.append(index-1)
#     else:
#         print("第%d个样本攻击失败了" % index)
#         print("-------------------")
#     if count == 20:
#         break
#
# ad_siganl_save = np.array(ad_siganl_save)
# count_index = np.array(count_index)
# source_signal = right_train[count_index]
#
# save_path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/Experimental Result/其他攻击方法/LSTM/MI-FGSM_7_to_4"
# if os.path.exists(save_path):   # 判断存储原信号的文件是否存在
#     pass
# else:                                   # 若不存在，则创建这个文件夹
#     os.makedirs(save_path)
# np.save(save_path+"/"+"限制噪声幅值0.1label_{}{}_source_signal.npy".format(right_y[0], category[right_y[0]]), source_signal)
# # 保存原信号
#
# # 计算对抗信号的信噪比
# save_snr = []
# for i in range(len(ad_siganl_save)):
#     example_ad = ad_siganl_save[i]
#     example_siganl = source_signal[i]
#
#     signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#     noise_of_signal = example_siganl - signal
#     snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#     # print(snr1)
#
#     noise_of_example_ad = example_ad - signal
#     snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#     save_snr.append(snr)
#
# result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))   # 对抗信号识别结果
# result_ad = np.argmax(result_ad, axis=1)
# # 创建excel表格
# workbook = xlwt.Workbook(encoding='ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# style = xlwt.XFStyle()  # 初始化样式
# font = xlwt.Font()  # 为样式创建字体
# font.name = 'Times New Roman'
# font.bold = True  # 黑体
# font.underline = True  # 下划线
# font.italic = True  # 斜体字
# style.font = font  # 设定样式
# worksheet.write(0, 0, '对抗样本序号')
# worksheet.write(0, 1, '识别结果')
# worksheet.write(0, 2, '攻击成功率')
# worksheet.write(0, 3, '对抗样本的信噪比')
# worksheet.write(1, 2, count / len(right_train))
#
# c = 0
# for r in result_ad:
#     print(int(r))
#     worksheet.write(c + 1, 0, c)  # 写入序号
#     worksheet.write(c + 1, 3, int(save_snr[c]))    # 写入对抗样本的信噪比
#     worksheet.write(c + 1, 1, int(r))  # 写入对抗样本的识别结果
#     c = c + 1
#
# workbook.save(save_path+"/"+'原类标为{}.xls'.format(right_y[0]))  # 保存文件
# np.save(save_path+"/"+"限制噪声幅值label_{}_ad_siganl_save.npy".format(right_y[0]), ad_siganl_save)