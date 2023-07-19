import tensorflow as tf

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import foolbox
import sys

import os
from keras.layers import Dense, Dropout, Flatten, LSTM, Input
from keras import backend as K
from keras.models import Model
import numpy as np
import foolbox
import keras.backend.tensorflow_backend as KTF
import xlwt
from foolbox.criteria import TargetClass
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 分配使用率
sess = tf.Session(config=config)
KTF.set_session(sess)


category = {0:"WBFM", 1:"QPSK", 2:"QAM64", 3: "QAM16", 4:"PAM4", 5:"GFSK", 6:"CPFSK" ,7:"BPSK", 8:"8PSK", 9:"AM-SSB", 10:"AM-DSB"}

'''模型搭建部分'''
input_x = Input(shape=(128, 2, ))
x = LSTM(units=128, return_sequences=True)(input_x)
x = LSTM(units=128, return_sequences=True)(x)
x = Flatten()(x)
pred = Dense(11, activation='softmax')(x)
model = Model(input_x, pred)
model.compile('adam', 'categorical_crossentropy', ['acc', ])
model.summary()
try:
    model.load_weights("/data0/quziwen/yelinhui/explainable/Deepsig.10A/LSTM_v/model_128_128_1.tfl")
    print("载入模型成功")
except:
    print("载入模型失败")


attack_name = ['CW', 'FGSM', 'LB-FGSM', 'JSMA', 'IBM', 'MI-FGSM', 'DeepFool',
               'PGD', 'DeepFoolL2',  # 0-8
               'DeepFoolLinf', 'AdditiveGaussian', 'SaltAndPepper', 'Boundary',
               'NewtonFool', 'RandomPGD']

DATA_PATH = '/data0/quziwen/yelinhui/explainable/dataset/'
testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')         # 44000 128 2
testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy')   # 2 44000
from foolbox.criteria import TargetClass

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


'''
目标攻击
'''
# label_to_attack = [7, 8]
# print(label_to_attack)
# target_label = {1: 7, 8: 1, 7: 4, 4: 7, 0: 5, 5: 0}
# for c in label_to_attack:
#     index = []
#     a = np.array([18, c], dtype=np.float64)   # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                      # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)                   # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 128, 2))   # 找出的需要攻击的信号
#     y_test = testSnrY[1]
#     y_test = y_test[index]                             # 找出的标签
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#
#     print("----processing data----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]     # 找到的所有预测正确的数据,这里只取了n个
#     right_y = y_test[right_index]        # 找到的所有预测正确的数据的类标
#     print("number of right sample", len(right_y))
#     right = model.predict(right_train)
#     right = np.argmax(right, axis=1)
#
#
#     count = 0         # 用来计算攻击成功的样本数
#     count_index = []  # 用来存放攻击成功的样本的索引
#     index = 0         # 用来打印哪一个样本攻击失败了
#     ad_siganl_save = []
#     for example in right_train:
#         index = index+1
#         fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
#         attack = foolbox.attacks.FGSM(fmodel, TargetClass(target_label[c]))  # 目标攻击
#         # attack = foolbox.attacks.FGSM(fmodel)                              # 无目标攻击
#         ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1, epsilons = 50)
#
#         if np.sum(ad_siganl) != None:
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#             # print("success")
#         else:
#             print(index, "failed")
#             print("-------------------")
#         # if count == 2:
#         #     break
#
#     # if len(ad_siganl_save) == 0:
#     #     continue
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#     path = "/home/yelinhui/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_LSTM/target_attack/label_{}_to_{}/".format(c, target_label[c])
#     os.makedirs(path, exist_ok=True)
#     ad_to_save_path = path + "label_{}_ad_siganl_save_{}.npy".format(c, target_label[c])
#
#     source_to_save_path = path + "label_{}_source_signal_{}.npy".format(c, target_label[c])
#
#     np.save(ad_to_save_path, ad_siganl_save)
#     np.save(source_to_save_path, source_signal)
#
#     acc = count/len(right_train)
#
#     result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))
#     result_ad = np.argmax(result_ad, axis=1)
#     with open(path+"label_{}_to_{}.txt".format(c, target_label[c]), "w") as f:
#         f.write(str(acc) + "\n")
#         for index, result in enumerate(result_ad):
#             label_save = str(index) + ":" + str(result) + "\n"
#             f.write(label_save)
#         f.close()


'''
无目标攻击
'''
# for c in range(6, 11):
#     print("Class", c)
#     index = []
#     a = np.array([18, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)    # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 128, 2))   # 找出的需要攻击的信号
#     y_test = testSnrY[1]
#     y_test = y_test[index]                             # 找出的标签
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#
#     print("----processing data----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]    # 找到的所有预测正确的数据,这里只取了n个
#     right_y = y_test[right_index]        # 找到的所有预测正确的数据的类标
#     print("number", len(right_y))
#     right = model.predict(right_train)   #
#     right = np.argmax(right, axis=1)
#
#     count = 0         # 用来计算攻击成功的样本数
#     count_index = []  # 用来存放攻击成功的样本的索引
#     index = 0         # 用来打印哪一个样本攻击失败了
#     ad_siganl_save = []
#     for example in right_train:
#         index = index+1
#         fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
#         # attack = foolbox.attacks.FGSM(fmodel, TargetClass(target_label[c]))  # 目标攻击
#         attack = foolbox.attacks.FGSM(fmodel)                              # 无目标攻击
#         ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1, epsilons = 50)
#
#         if np.sum(ad_siganl) != None:
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#             # print("success")
#         else:
#             print(index, "failed")
#             print("-------------------")
#
#         # if count == 2:
#         #     break
#
#     if len(ad_siganl_save) == 0:
#         continue
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#     path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_LSTM/un_target_attack/label_{}/".format(c)
#     os.makedirs(path, exist_ok=True)
#
#     ad_to_save_path = path + "label_{}_ad_siganl_save.npy".format(c)
#     source_to_save_path = path + "label_{}_source_signal.npy".format(c)
#
#     np.save(ad_to_save_path, ad_siganl_save)
#     np.save(source_to_save_path, source_signal)
#     acc = count/len(right_train)
#
#     result_ad = model.predict(ad_siganl_save.reshape(-1, 128, 2))
#     result_ad = np.argmax(result_ad, axis=1)
#     with open(path+"label_{}.txt".format(c), "w") as f:
#         f.write(str(acc) + "\n")
#         for index, result in enumerate(result_ad):
#             label_save = str(index) + ":" + str(result) + "\n"
#             f.write(label_save)
#         f.close()

'''
画图部分
'''
# def conv_out(model, layer_name):
#     for l in model.layers:
#         if l.name == layer_name:
#             return l.output
#
# source_signal = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_LSTM/target_attack/label_0_to_5/label_0_source_signal_5.npy")
# ad_signal = np.load("/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1/不限制噪声幅值label_7_ad_siganl_save.npy")
#
# layername = "lstm_1"
# count = 0
# for img in source_signal:
#     img = img.reshape(1, 128, 2)
#     model_1 = Model(inputs=model.input, outputs=conv_out(model, layername))
#     loss = K.sum(model_1.output, axis=1)  # LSTM层的输出的每一列求和的输出
#     lstm_sum_output = K.function([model_1.input], [loss])  # 计算
#     out = np.array(lstm_sum_output([img])).reshape(128)
#     index = np.argsort(out)[-5:]
#     index_loss1 = index[0]
#     index_loss2 = index[1]
#     index_loss3 = index[2]
#     index_loss4 = index[3]
#     index_loss5 = index[4]
#     # index_loss6 = index[5]
#     # index_loss7 = index[6]
#     # index_loss8 = index[7]
#     # index_loss9 = index[8]
#     # index_loss10 = index[9]
#     # index_loss11 = index[10]
#     # index_loss12 = index[11]
#     # index_loss13 = index[12]
#     # index_loss14 = index[13]
#     # index_loss15 = index[14]
#     # index_loss16 = index[15]
#     # index_loss17 = index[16]
#     # index_loss18 = index[17]
#     # index_loss19 = index[18]
#     # index_loss20 = index[19]
#     # index_loss21 = index[20]
#     # index_loss22 = index[21]
#     # index_loss23 = index[22]
#     # index_loss24 = index[23]
#     # index_loss25 = index[24]
#     # index_loss26 = index[25]
#     # index_loss27 = index[26]
#     # index_loss28 = index[27]
#     # index_loss29 = index[28]
#     # index_loss30 = index[29]
#
#     model_1 = Model(inputs=model.input, outputs=conv_out(model, layername))
#     loss = K.sum(model_1.output, axis=1)  # LSTM层输出按列就和后作为输出
#     loss_sum = loss[:, index_loss1: index_loss1 + 1] + loss[:, index_loss2: index_loss2 + 1] + \
#                loss[:, index_loss3: index_loss3 + 1] + loss[:, index_loss4: index_loss4 + 1] + \
#                loss[:, index_loss5: index_loss5 + 1]
#                # loss[:, index_loss7: index_loss7 + 1] + loss[:, index_loss8: index_loss8 + 1] + \
#                # loss[:, index_loss9: index_loss9 + 1] + loss[:, index_loss10: index_loss10 + 1] + \
#                # loss[:, index_loss11: index_loss11 + 1] + loss[:, index_loss12: index_loss12 + 1] + \
#                # loss[:, index_loss13: index_loss13 + 1] + loss[:, index_loss14: index_loss14 + 1] + \
#                # loss[:, index_loss15: index_loss15 + 1] + loss[:, index_loss18: index_loss16 + 1] + \
#                # loss[:, index_loss17: index_loss17 + 1] + loss[:, index_loss18: index_loss18 + 1] + \
#                # loss[:, index_loss19: index_loss19 + 1] + loss[:, index_loss20: index_loss20 + 1]
#                 # loss[:, index_loss21: index_loss21 + 1] + loss[:, index_loss22: index_loss22 + 1] + \
#                 # loss[:, index_loss23: index_loss23 + 1] + loss[:, index_loss24: index_loss24 + 1] + \
#                 # loss[:, index_loss25: index_loss25 + 1] + loss[:, index_loss26: index_loss26 + 1] + \
#                 # loss[:, index_loss27: index_loss27 + 1] + loss[:, index_loss28: index_loss28 + 1] + \
#                 # loss[:, index_loss29: index_loss29 + 1] + loss[:, index_loss30: index_loss30 + 1]
#     # 选择每一类的目标神经元索引，将目标神经元的激活值求和后最为loss反向求梯度作为特征矩阵
#
#     grads = K.gradients(loss_sum, model_1.input)[0]  # 选中的神经元激活值之和对输入求导
#     feature = K.function([model_1.input], [grads])  # 计算LSTM层选中的神经元的输出之和对输入的导数
#     feature_out = feature([img])
#     feature_out = np.array(feature_out).reshape(128, 2)
#
#     output_path_bx = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1/source_signal/波形/"
#     if os.path.exists(output_path_bx):
#         pass
#     else:
#         os.makedirs(output_path_bx)
#
#     output_path_xz = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1/source_signal/星座/"
#     if os.path.exists(output_path_xz):
#         pass
#     else:
#         os.makedirs(output_path_xz)
#
#     '''画波形激活图'''
#     e = img.reshape(128, 2)
#     first_a = e[0:128, 0]
#     last_a = e[0:128, 1]
#     plt.figure(figsize=(13, 7))
#     plt.ion()
#     plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=feature_out[:, 0].reshape(128), s=100)
#     plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=feature_out[:, 1].reshape(128), s=100)
#     file_output = output_path_bx + "/" + "原信号波形"+ str(count) + ".png"
#     plt.colorbar()
#     plt.savefig(file_output, format='png', dpi=500, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
#     '''星座激活图'''
#     camweight = feature_out[:, 0]+feature_out[:, 1]
#     camweight = camweight/2
#     e = img.reshape(128, 2)
#     first_a = e[0:128, 0]
#     last_a = e[0:128, 1]
#     plt.figure(figsize=(7, 7))
#     plt.xlim(-1.2, 1.2)
#     plt.ylim(-1.2, 1.2)
#     plt.grid(linestyle='-.')
#     file_output = output_path_xz + "/" + "原信号星座" +str(count) +".png"
#     plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=camweight.reshape(128), s=100)
#     plt.colorbar()
#     plt.savefig(file_output, format='png', dpi=500, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#     count = count+1
#     if count == 30:
#         break
#
# count = 0
# for img in ad_signal:
#     img = img.reshape(1, 128, 2)
#     model_1 = Model(inputs=model.input, outputs=conv_out(model, layername))
#     loss = K.sum(model_1.output, axis=1)  # LSTM层的输出的每一列求和的输出
#     lstm_sum_output = K.function([model_1.input], [loss])  # 计算
#     out = np.array(lstm_sum_output([img])).reshape(128)
#     index = np.argsort(out)[-5:]
#     index_loss1 = index[0]
#     index_loss2 = index[1]
#     index_loss3 = index[2]
#     index_loss4 = index[3]
#     index_loss5 = index[4]
#     # index_loss6 = index[5]
#     # index_loss7 = index[6]
#     # index_loss8 = index[7]
#     # index_loss9 = index[8]
#     # index_loss10 = index[9]
#     # index_loss11 = index[10]
#     # index_loss12 = index[11]
#     # index_loss13 = index[12]
#     # index_loss14 = index[13]
#     # index_loss15 = index[14]
#     # index_loss16 = index[15]
#     # index_loss17 = index[16]
#     # index_loss18 = index[17]
#     # index_loss19 = index[18]
#     # index_loss20 = index[19]
#     # index_loss21 = index[20]
#     # index_loss22 = index[21]
#     # index_loss23 = index[22]
#     # index_loss24 = index[23]
#     # index_loss25 = index[24]
#     # index_loss26 = index[25]
#     # index_loss27 = index[26]
#     # index_loss28 = index[27]
#     # index_loss29 = index[28]
#     # index_loss30 = index[29]
#
#     model_1 = Model(inputs=model.input, outputs=conv_out(model, layername))
#     loss = K.sum(model_1.output, axis=1)  # LSTM层输出按列就和后作为输出
#     loss_sum = loss[:, index_loss1: index_loss1 + 1] + loss[:, index_loss2: index_loss2 + 1] + \
#                loss[:, index_loss3: index_loss3 + 1] + loss[:, index_loss4: index_loss4 + 1] + \
#                loss[:, index_loss5: index_loss5 + 1]
#                # loss[:, index_loss7: index_loss7 + 1] + loss[:, index_loss8: index_loss8 + 1] + \
#                # loss[:, index_loss9: index_loss9 + 1] + loss[:, index_loss10: index_loss10 + 1] + \
#                # loss[:, index_loss11: index_loss11 + 1] + loss[:, index_loss12: index_loss12 + 1] + \
#                # loss[:, index_loss13: index_loss13 + 1] + loss[:, index_loss14: index_loss14 + 1] + \
#                # loss[:, index_loss15: index_loss15 + 1] + loss[:, index_loss18: index_loss16 + 1] + \
#                # loss[:, index_loss17: index_loss17 + 1] + loss[:, index_loss18: index_loss18 + 1] + \
#                # loss[:, index_loss19: index_loss19 + 1] + loss[:, index_loss20: index_loss20 + 1]
#                 # loss[:, index_loss21: index_loss21 + 1] + loss[:, index_loss22: index_loss22 + 1] + \
#                 # loss[:, index_loss23: index_loss23 + 1] + loss[:, index_loss24: index_loss24 + 1] + \
#                 # loss[:, index_loss25: index_loss25 + 1] + loss[:, index_loss26: index_loss26 + 1] + \
#                 # loss[:, index_loss27: index_loss27 + 1] + loss[:, index_loss28: index_loss28 + 1] + \
#                 # loss[:, index_loss29: index_loss29 + 1] + loss[:, index_loss30: index_loss30 + 1]
#     # 选择每一类的目标神经元索引，将目标神经元的激活值求和后最为loss反向求梯度作为特征矩阵
#
#     grads = K.gradients(loss_sum, model_1.input)[0]  # 选中的神经元激活值之和对输入求导
#     feature = K.function([model_1.input], [grads])  # 计算LSTM层选中的神经元的输出之和对输入的导数
#     feature_out = feature([img])
#     feature_out = np.array(feature_out).reshape(128, 2)
#
#     output_path_bx = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1/ad_signal/波形/"
#     if os.path.exists(output_path_bx):
#         pass
#     else:
#         os.makedirs(output_path_bx)
#
#     output_path_xz = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/对抗攻击部分/7_to_4_限制幅值为0.1/ad_signal/星座/"
#     if os.path.exists(output_path_xz):
#         pass
#     else:
#         os.makedirs(output_path_xz)
#
#     '''画波形激活图'''
#     e = img.reshape(128, 2)
#     first_a = e[0:128, 0]
#     last_a = e[0:128, 1]
#     plt.figure(figsize=(13, 7))
#     plt.ion()
#     plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=feature_out[:, 0].reshape(128), s=100)
#     plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=feature_out[:, 1].reshape(128), s=100)
#     file_output = output_path_bx + "/" + "对抗信号波形"+ str(count) + ".png"
#     plt.colorbar()
#     plt.savefig(file_output, format='png', dpi=500, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
#     '''星座激活图'''
#     camweight = feature_out[:, 0]+feature_out[:, 1]
#     camweight = camweight/2
#     e = img.reshape(128, 2)
#     first_a = e[0:128, 0]
#     last_a = e[0:128, 1]
#     plt.figure(figsize=(7, 7))
#     plt.xlim(-1.2, 1.2)
#     plt.ylim(-1.2, 1.2)
#     plt.grid(linestyle='-.')
#     file_output = output_path_xz + "/" + "对抗信号星座" +str(count) +".png"
#     plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=camweight.reshape(128), s=100)
#     plt.colorbar()
#     plt.savefig(file_output, format='png', dpi=500, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#     count = count+1
#     if count == 30:
#         break



'''
攻击用于对抗训练
'''
# DATA_PATH = '/data0/quziwen/yelinhui/explainable/dataset/'
# trainX = np.load(DATA_PATH + 'radio11CNormTrainX.npy')         # 44000 128 2
# trainSnrY = np.load(DATA_PATH + 'radio11CNormTrainSnrY.npy')   # 2 44000
#
# import random
# snrs = [12, 14, 16, 18]
# for c in range(0, 11):
#     index = []
#     for snr in snrs:
#         a = np.array([snr, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#         count = 0
#         for i in trainSnrY.T:                         # 找所需要测试的信号的索引
#             if i[0] == a[0] and i[1] == a[1]:                          # 找到对应类标的索引
#                 index.append(count)
#             count = count+1
#     index = np.array(index)
#     train = trainX[index]       # 训练集中对应类标的样本
#     train = np.reshape(train, (len(train), 128, 2))   # 找出的需要攻击的信号
#     y_train = trainSnrY[:, index]    # 目标类标的标签
#
#     list_val = np.arange(len(train))  # list_val==array([0,1,2,,...,44000])   len(testX)==44000
#     random.shuffle(list_val)  # 将序列的所有元素随机排序
#     train = train[list_val]
#     y_train = y_train[:, list_val]
#
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#     print(result.shape)
#     print("----processing data----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#     right_index = np.array(right_index)  # 找到的所有预测正确的数据的索引
#     right_train = train[right_index]    # 找到的所有预测正确的数据,这里只取了n个
#     right_y = y_train[:, right_index]        # 找到的所有预测正确的数据的类标
#     print("number", right_y.shape[1])
#
#     count = 0         # 用来计算攻击成功的样本数
#     count_index = []  # 用来存放攻击成功的样本的索引
#     index = 0         # 用来打印哪一个样本攻击失败了
#     ad_siganl_save = []
#     for example in right_train:
#         index = index+1
#         fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), predicts='probabilities')
#         attack = foolbox.attacks.FGSM(fmodel)                              # 无目标攻击
#         ad_siganl = attack(example, label=c, max_epsilon=0.15, epsilons = 15)
#
#         if np.sum(ad_siganl) != None:
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#             print("success")
#         else:
#             print(index, "failed")
#             print("-------------------")
#
#         if count == 600:
#             break
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#     source_label_of_ad = right_y[:, count_index]
#     path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_LSTM/无目标攻击--用于对抗训练/label_{}/".format(c)
#     os.makedirs(path, exist_ok=True)
#
#     ad_to_save_path = path + "label_{}_ad_siganl_save.npy".format(c)
#     source_to_save_path = path + "label_{}_source_signal.npy".format(c)
#     label_to_save_path = path + "source_{}_label_of_ad.npy".format(c)
#
#     np.save(ad_to_save_path, ad_siganl_save)
#     np.save(source_to_save_path, source_signal)
#     np.save(label_to_save_path, source_label_of_ad)