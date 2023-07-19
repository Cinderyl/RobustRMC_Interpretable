# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.ion()
import math
import time
import os
import numpy as np
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
number_of_point = 100
import seaborn as sns
model_name = {0:"LSTM128", 1:"LSTM256", 2:"Alexnet", 3:"Resnet", 4:"NIN", 5:"LSTM64", 6:"LSTM128_64", 7:"LSTM128_128", 8:"Vgg16",9:"LSTM32", 10:"LSTM128_32"}
# sns.set(style="darkgrid")

'''
画各个模型在各个信噪比下对各个类的识别准确率
'''
# result = np.load("/home/NewDisk/yelinhui/explainable/Deepsig.10A/NIN/mytest/用幅值进行训练/对各个类信号的识别结果.npy")
# path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/识别准确率/各个信噪比下模型对各个类的识别准确率/"
# file_jihe = os.listdir(path)
# count = 0
# for file in file_jihe:
#     file_path = path+file
#     print(file_path)
#     result = np.load(file_path)
#     x = np.arange(-20, 20, 2)
#     # x = [1, 2.8, 4.6, 6.4, 8.2, 10]
#     x = list(x)
#     '''plot data'''
#     fig, ax = plt.subplots(figsize=(15, 7))
#     plt.rcParams['savefig.dpi'] = 1000       # 图片像素
#     plt.rcParams['figure.dpi'] = 1000        # 分辨率
#     # ax.semilogx(x, I, marker='o', mec='r', mfc='w', label='SVM模型')
#     # ax.semilogx(x, II, marker='o', mec='r', mfc='w', label='卷积神经网络')
#     ax.plot(x, result[:, 0],  "--",color = 'peru', marker='o', mec='r', mfc='w', label='WBFM')
#     ax.plot(x, result[:, 1], ":",color = 'dodgerblue', marker='^', mec='lightcoral', mfc='w', label='QPSK')
#     ax.plot(x, result[:, 2], "-.",color = 'purple', marker='s', mec='y', mfc='w', label='QAM64')
#     ax.plot(x, result[:, 3], '--',color = 'brown', marker='+', mec='r', mfc='w', label='QAM16')
#     ax.plot(x, result[:, 4], "-.",color = 'orange', marker='^', mec='r', mfc='w', label='PAM4')
#     ax.plot(x, result[:, 5], "-.", color='y', marker='s', mec='y', mfc='w',label='GFSK')
#     ax.plot(x, result[:, 6], "--", color='black', marker='o', mec='r', mfc='w', label='CPFSK')
#     ax.plot(x, result[:, 7], "-",color = 'r', marker='*', mec='r', mfc='w', label='BPSK')
#     ax.plot(x, result[:, 8], "+-",color = 'darkcyan', marker='*', mec='r', mfc='w', label='8PSK')
#     ax.plot(x, result[:, 9], "x:",color = 'darkgoldenrod', marker='*', mec='r', mfc='w', label='AM-SSB')
#     ax.plot(x, result[:, 10], "d-",color = 'mediumblue', marker='*', mec='r', mfc='w', label='AM-DSB')
#     ax.set_xlabel('Signal to noise ration(dB)', fontsize = 20)
#     ax.set_ylabel('Accuracy of recognition', fontsize = 20)
#     # plt.xlabel('poison percentage')  # X轴标签
#     # plt.ylabel("Attack success rate(%)")  # Y轴标签
#     ax.tick_params(labelsize=15)            # 刻度字体大小
#     # box = ax.get_position()
#     # ax.set_position([box.x0, box.y0, box.width*0.5, box.height* 0.8 ])
#     fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2),frameon=True,ncol=1)
#     fig.set_tight_layout(tight='rect')
#     # plt.title(name,y=-0.25)
#     plt.savefig("/home/NewDisk/yelinhui/explainable/Deepsig.10A/识别准确率/{}模型识别准确率.png".format(model_name[count]), format='png', dpi=500, bbox_inches='tight')
#     plt.close()
#     count = count+1


'''
画模型在各个信噪比下的识别准确率
'''
NIN = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/各个模型的准确率/NIN.npy")
VGG16 = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/各个模型的准确率/VGG16.npy")
Alexnet = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/各个模型的准确率/Alexnet.npy")
LSTM = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/各个模型的准确率/LSTM.npy")
model_hubu = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/各个模型的准确率/VGG16+LSTM_accuracy.npy")

# path = "/home/yelinhui/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 model_hubu/New Folder/"
# file_jihe = os.listdir(path)
# example = []
# for file in file_jihe:
#     print(file)
#     file_path = path+file
#     a = np.load(file_path)
#     example.append(a)
# #
# example = np.array(example)
x = np.arange(-20, 20, 2)
x = list(x)
'''plot data'''
fig, ax = plt.subplots(figsize=(12, 7))
plt.rcParams['savefig.dpi'] = 1000       # 图片像素
plt.rcParams['figure.dpi'] = 1000        # 分辨率
ax.yaxis.grid(color='black',
              linestyle=':',
              linewidth=1.5,
              alpha=0.3)
# ax.plot(x, example[0, :],  "--",color = 'peru', marker='o', mec='r', mfc='w', label='LSTM128')
# ax.plot(x, example[1, :], ":",color = 'dodgerblue', marker='^', mec='lightcoral', mfc='w', label='Resnet')
# ax.plot(x, model_hubu, "x:", color='dimgrey', marker='*', markersize=10, linewidth=3, mec='r', mfc='w', label='VGG16+LSTM')
ax.plot(x, NIN, "-.", color='purple', markersize=10, linewidth=3, marker='^', mec='y', mfc='w', label='NIN')
# ax.plot(x, example[4, :], '--',color = 'brown', marker='+', mec='r', mfc='w', label='LSTM_256')
# ax.plot(x, example[5, :], "-.",color = 'orange', marker='^', mec='r', mfc='w', label='LSTM128_32')
ax.plot(x, LSTM, "-.", color='y', markersize=10, linewidth=3, marker='s', mec='y', mfc='w', label='LSTM')
# ax.plot(x, example[7, :], "--", color='black', marker='o', mec='r', mfc='w', label='LSTM32')
ax.plot(x, VGG16, "-", color='r', markersize=10, linewidth=3, marker='o', mec='r', mfc='w', label='VGG16')   # 8
ax.plot(x, Alexnet, "+-", color='darkcyan',  markersize=10, linewidth=3, marker='*', mec='r', mfc='w', label='Alexnet')
# ax.plot(x, example[10, :], "x:",color = 'darkgoldenrod', marker='*', mec='r', mfc='w', label='LSTM64')
# ax.plot(x, example[11, :], "d-",color = 'mediumblue', marker='*', mec='r', mfc='w', label='LSTM128_64')
# ax.set_xlabel('信噪比(dB)', fontsize=24)
# ax.set_ylabel('识别准确率', fontsize=24)
ax.tick_params(labelsize=20)            # 刻度字体大小
fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2), frameon=True, ncol=1, fontsize = 24)
fig.set_tight_layout(tight='rect')
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.savefig("/data0/quziwen/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 不同分类器提取的特征的差异+模型互补/4model_accuracy12比7中文版.png", format='png', dpi=500, bbox_inches='tight')
plt.show()
plt.close()


'''
画单个模型在各个信噪比下对各个类的识别准确率
'''
# result = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/NIN/mytest/model_128_2/NIN_对各个类信号的识别准确率.npy")
# x = np.arange(-20, 20, 2)
# x = list(x)
# '''plot data'''
# fig, ax = plt.subplots(figsize=(15, 7))
# plt.rcParams['savefig.dpi'] = 1000       # 图片像素
# plt.rcParams['figure.dpi'] = 1000        # 分辨率
# # ax.semilogx(x, I, marker='o', mec='r', mfc='w', label='SVM模型')
# # ax.semilogx(x, II, marker='o', mec='r', mfc='w', label='卷积神经网络')
# ax.plot(x, result[:, 0],  "--",color = 'peru', marker='o', mec='r', mfc='w', label='WBFM')
# ax.plot(x, result[:, 1], ":",color = 'dodgerblue', marker='^', mec='lightcoral', mfc='w', label='QPSK')
# ax.plot(x, result[:, 2], "-.",color = 'purple', marker='s', mec='y', mfc='w', label='QAM64')
# ax.plot(x, result[:, 3], '--',color = 'brown', marker='+', mec='r', mfc='w', label='QAM16')
# ax.plot(x, result[:, 4], "-.",color = 'orange', marker='^', mec='r', mfc='w', label='PAM4')
# ax.plot(x, result[:, 5], "-.", color='y', marker='s', mec='y', mfc='w',label='GFSK')
# ax.plot(x, result[:, 6], "--", color='black', marker='o', mec='r', mfc='w', label='CPFSK')
# ax.plot(x, result[:, 7], "-",color = 'r', marker='*', mec='r', mfc='w', label='BPSK')
# ax.plot(x, result[:, 8], "+-",color = 'darkcyan', marker='*', mec='r', mfc='w', label='8PSK')
# ax.plot(x, result[:, 9], "x:",color = 'darkgoldenrod', marker='*', mec='r', mfc='w', label='AM-SSB')
# ax.plot(x, result[:, 10], "d-",color = 'mediumblue', marker='*', mec='r', mfc='w', label='AM-DSB')
# ax.set_xlabel('Signal to noise ration(dB)', fontsize = 20)
# ax.set_ylabel('Accuracy of recognition', fontsize = 20)
# # plt.xlabel('poison percentage')  # X轴标签
# # plt.ylabel("Attack success rate(%)")  # Y轴标签
# ax.tick_params(labelsize=15)            # 刻度字体大小
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width*0.5, box.height* 0.8 ])
# fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2),frameon=True,ncol=1)
# fig.set_tight_layout(tight='rect')
# # plt.title(name,y=-0.25)
# # plt.savefig("/home/NewDisk/yelinhui/explainable/Deepsig.10A/识别准确率/{}模型识别准确率.png".format("Resnet"), format='png', dpi=500, bbox_inches='tight')
# plt.show()
# # plt.close()

'''
画单个模型在各个信噪比下的识别准确率
'''
# result = np.load("/home/NewDisk/yelinhui/explainable/Deepsig.10A/LSTM特征可视化/LSTM模型在各个信噪比下的识别准确率--对抗.npy")
# example = np.array(result)
#
# result_normal = np.load("/home/NewDisk/yelinhui/explainable/Deepsig.10A/模型互补/NIN模型各个信噪比下的识别准确率.npy")
# example_normal = np.array(result_normal)
#
# x = np.arange(-20, 20, 2)
# x = list(x)
# fig, ax = plt.subplots(figsize=(15, 7))
# plt.rcParams['savefig.dpi'] = 1000       # 图片像素
# plt.rcParams['figure.dpi'] = 1000        # 分辨率
# # ax.plot(x, example, "--", color = 'peru', marker='o', mec='r', mfc='w', label='LSTM128--对抗')
# ax.plot(x, example_normal, "-", color = 'r', marker='o', mec='r', mfc='w', label='LSTM128--对抗')
# fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2), frameon=True, ncol=1)
# plt.show()
