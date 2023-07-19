import numpy as np
import os, time
import itertools
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = {0:"LSTM128", 1:"LSTM256", 2:"Alexnet", 3:"Lstm128_128", 4:"Lstm128_64", 5:"Lstm128_32", 6:"Vgg16", 7:"Lstm32", 8:"NIN", 9:"Resnet50", 10:"Lstm64"}
"根据混淆矩阵的具体数值画混淆矩阵的热力图"
def plot_confusion_matrix(cm, labels_name, normalize = True):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    # plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #
    # if normalize:
    #     thresh = cm.max() / 1.5
    # else:
    #     thresh = cm.max() / 2
    #
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     if normalize:
    #         plt.text(j, i, "{:0.2f}".format(cm[i, j]),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "white")
    #     else:
    #         plt.text(j, i, "{:,}".format(cm[i, j]),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "white")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


labels_name = ["WBFM", "QPSK", "QAM64", "QAM16", "PAM4", "GFSK", "CPFSK", "BPSK", "8PSK", "AM-SSB", "AM-DSB"]
import matplotlib.pyplot as plt
# path = "/home/NewDisk/yelinhui/explainable/Deepsig.10A/混淆矩阵/"
# file_jihe = os.listdir(path)
# count = 0
# 多个画图图
# for file in file_jihe:
#     file_path = path+file
#     print(file_path)
#     confuse_matric = np.load(file_path)
#     print(confuse_matric.shape)
#     plot_confusion_matrix(confuse_matric, labels_name)
#     plt.savefig("/home/NewDisk/yelinhui/explainable/Deepsig.10A/混淆矩阵/{}模型混淆矩阵.png".format(model_name[count]), format='png', dpi=500, bbox_inches='tight')
#     plt.close()
#     # plt.show()
#     count = count+1

'''
单一画图
'''
confuse_matric = np.load("/data0/quziwen/yelinhui/explainable/Deepsig.10A/LSTM_v/CM.npy")
print(confuse_matric)
# confuse_matric[]

plot_confusion_matrix(confuse_matric, labels_name)
# plt.savefig("/home/yelinhui/yelinhui/explainable/Deepsig.10A/paper_result/4.3 Visualization classifier/4.3.2 model_hubu/model_hubu.png", format='png', dpi=500, bbox_inches='tight')
plt.show()
