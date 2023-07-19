import numpy as np
import random
import os
from os.path import join, basename

DATA_PATH = '/data0/liaodanxin/explainable/dataset/'

trainX = np.load(DATA_PATH + 'radio11CNormTrainX.npy')
trainSnrY = np.load(DATA_PATH + 'radio11CNormTrainSnrY.npy')
trainSnr = trainSnrY[0]
trainY = trainSnrY[1]


testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')
testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy')
testSnr = testSnrY[0]
testY = testSnrY[1]

path = "/home/liaodanxin/explainable/Deepsig.10A/NIN/对抗训练/delate/test_ad"
files = os.listdir(path)

path_source = "/home/NewDisk/liaodanxin/explainable/Deepsig.10A/NIN/对抗训练/delate/test_source"
files_sources = os.listdir(path_source)


test_ad = np.zeros((1, 16, 16))
for file in files:
    file_path = path+"/"+file
    file_ad = np.load(file_path)
    test_ad = np.concatenate((test_ad, file_ad), axis=0)
test_ad = np.array(test_ad, dtype=np.float32)
np.save("/home/NewDisk/liaodanxin/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_ad.npy", test_ad[1:])

for file in files:
    number = file[12:-19]
    for files_source in files_sources:
        number_s = files_source[12:-18]
        if number == number_s:
            file_s = np.load(path_source+"/" + files_source)
            test_ad = np.concatenate((test_ad, file_s), axis=0)
test_ad = np.array(test_ad, dtype=np.float32)
np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_source.npy", test_ad[1:])


# ad_snrY = []
# for file in files:
#     number = file[12:-19]
#     for files_source in files_sources:
#         number_s = files_source[12:-18]
#         if number == number_s:
#             file_s = np.load(path_source+"/"+ files_source)
#             file_ad = np.load(path +"/" +file)
#             for i in range(len(file_ad)):
#                 example_ad = file_ad[i]
#                 example_siganl = file_s[i]
#
#                 signal = example_siganl * (10 ** 0.9 / (10 ** 0.9 + 1))
#                 noise_of_signal = example_siganl - signal
#                 snr1 = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_signal ** 2))
#                 # print(snr1)
#
#                 noise_of_example_ad = example_ad - signal
#                 snr = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise_of_example_ad ** 2))
#
#                 snr = int(snr)
#                 snr = int((snr+1)/2)*2
#
#                 number = int(number)
#                 example_snrY = [snr, number]
#                 ad_snrY.append(example_snrY)  # 保存对抗样本的信噪比和类标
#
# ad_snrY = np.array(ad_snrY).T
# print(ad_snrY.shape)
# # trainSnrY = np.concatenate((trainSnrY, ad_snrY), axis=1)
# np.save("/home/NewDisk/yelinhui/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_ad_SnrY.npy", ad_snrY)
#
#
#
