# -*- coding: utf-8 -*-
import pickle
import numpy as np

'''
这个py处理方式后的数据拿来训练是可以的，但是测试缪盛欢训练好的模型时，模型输出的数据是不对的
因为这个py文件生成的IQ信号处理方式与缪盛欢训练时是不同的,原因是因为没有正则化
'''

# word = pickle.load(open("/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.10A/RML2016.10a_dict.pkl", 'rb'), encoding='iso-8859-1')
word = pickle.load(open("/data0/quziwen/yelinhui/explainable/dataset/RML2016.10a_dict.pkl", 'rb'), encoding='iso-8859-1')

radio11CNormTrainX = np.zeros((1, 128, 2))
radio11CNormTrainSnrY = np.zeros((2, 1))
radio11CNormTestX = np.zeros((1, 128, 2))
radio11CNormTestSnrY = np.zeros((2, 1))

print(word["8PSK", 18].shape)

for key in word:
    if key[0]=="8PSK":
        snr = np.full((1, 1000), key[1])     # 信噪比
        savekey = np.full((1, 1000), 0)      # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)


    if key[0]=="AM-DSB":
        snr = np.full((1, 1000), key[1])     # 信噪比
        savekey = np.full((1, 1000), 1)      # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0]=="AM-SSB":
        snr = np.full((1, 1000), key[1])     # 信噪比
        savekey = np.full((1, 1000), 2)      # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "BPSK":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 3)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "CPFSK":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 4)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "GFSK":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 5)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "PAM4":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 6)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "QAM16":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 7)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "QAM64":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 8)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "QPSK":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 9)   # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)

    if key[0] == "WBFM":
        snr = np.full((1, 1000), key[1])  # 信噪比
        savekey = np.full((1, 1000), 10)  # 类标
        Y_train = np.concatenate((snr[:, 0:800], savekey[:, 0:800]), axis=0)
        data_train = word[key][0:800].reshape(800, 128, 2)                     # 1000,2,128

        Y_test = np.concatenate((snr[:, 800:1000], savekey[:, 800:1000]), axis=0)
        data_test = word[key][800:1000].reshape(200, 128, 2)

        radio11CNormTrainSnrY = np.concatenate((radio11CNormTrainSnrY, Y_train), axis=1)
        radio11CNormTrainX = np.concatenate((radio11CNormTrainX, data_train), axis=0)
        radio11CNormTestX = np.concatenate((radio11CNormTestX, data_test), axis=0)
        radio11CNormTestSnrY = np.concatenate((radio11CNormTestSnrY, Y_test), axis=1)


# radio11CNormTrainSnrY = radio11CNormTrainSnrY[1:220001]
# radio11CNormTrainX = radio11CNormTrainX[1:220001]
#
# radio11CNormTrainSnrYsave = radio11CNormTrainSnrY[0:176000]
# radio11CNormTrainXsave = radio11CNormTrainX[0:176000]
#
# radio11CNormTestSnrYsave = radio11CNormTrainSnrY[176000:220000]
# radio11CNormTestXsave = radio11CNormTrainX[176000:220000]

radio11CNormTrainSnrY1111 = radio11CNormTrainSnrY[:, 1: 176001]
radio11CNormTrainX1111 = radio11CNormTrainX[1: 176001]
radio11CNormTestSnrY1111 = radio11CNormTestSnrY[:, 1:44001]
radio11CNormTestX1111 = radio11CNormTestX[1:44001]
#
# print(radio11CNormTrainSnrY[:, 1:176001].shape)
# print(radio11CNormTrainX[1: 176001].shape)
# print(radio11CNormTestSnrY[:, 1:44001].shape)
# print(radio11CNormTestX[1:44001].shape)

# np.save("/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.10A/radio11CNormTrainSnrY1111.npy", radio11CNormTrainSnrY1111)
# np.save("/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.10A/radio11CNormTrainX1111.npy", radio11CNormTrainX1111)
#
# np.save("/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.10A/radio11CNormTestSnrY1111.npy", radio11CNormTestSnrY1111)
# np.save("/home/NewDisk/yelinhui/explainable/Dataset/Deepsig.10A/radio11CNormTestX1111.npy",radio11CNormTestX1111)


np.save("/data0/benke/ldx/explainable/dataset/read_data/radio11CNormTrainSnrY1111.npy", radio11CNormTrainSnrY1111)
np.save("/data0/benke/ldx/explainable/dataset/read_data/radio11CNormTrainX1111.npy", radio11CNormTrainX1111)

np.save("/data0/benke/ldx/explainable/dataset/read_data/radio11CNormTestSnrY1111.npy", radio11CNormTestSnrY1111)
np.save("/data0/benke/ldx/explainable/dataset/read_data/radio11CNormTestX1111.npy",radio11CNormTestX1111)