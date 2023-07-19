import numpy as np
import pickle

DATA = '/data0/quziwen/yelinhui/explainable/dataset/'
FILE_NAME = 'RML2016.10a_dict.pkl'

fr = open(DATA+FILE_NAME, 'rb')
signal = pickle.load(fr, encoding='iso-8859-1')

train_num = 800
test_num = 200
val_num = 50

dict_name = ['WBFM', 'QPSK', 'QAM64', 'QAM16', 'PAM4', 'GFSK', 'CPFSK', 'BPSK', '8PSK', 'AM-SSB', 'AM-DSB']
SNR_level = [18, 16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20]

Xtrain = np.zeros(shape=[len(dict_name)*len(SNR_level)*train_num, 128, 2],dtype=np.float32)
Ytrain = np.zeros(shape=[len(dict_name)*len(SNR_level)*train_num], dtype=np.int64)
SNRtrain = np.zeros(shape=[len(dict_name)*len(SNR_level)*train_num], dtype=np.int64)

Xtest = np.zeros(shape=[len(dict_name)*len(SNR_level)*test_num, 128, 2],dtype=np.float32)
Ytest = np.zeros(shape=[len(dict_name)*len(SNR_level)*test_num], dtype=np.int64)
SNRtest = np.zeros(shape=[len(dict_name)*len(SNR_level)*test_num], dtype=np.int64)

Xval = np.zeros(shape=[len(dict_name)*len(SNR_level)*val_num, 128, 2],dtype=np.float32)
Yval = np.zeros(shape=[len(dict_name)*len(SNR_level)*val_num], dtype=np.int64)
SNRval = np.zeros(shape=[len(dict_name)*len(SNR_level)*val_num], dtype=np.int64)

num_x = 0
label_signal = 0
for i in dict_name:
    for j in SNR_level:
        tmp_signal = np.transpose(a=signal[i, j], axes=[0, 2, 1])

        Xtrain[num_x*train_num : (num_x+1)*train_num] = tmp_signal[0:train_num]
        Ytrain[num_x*train_num : (num_x+1)*train_num] = label_signal
        SNRtrain[num_x*train_num : (num_x+1)*train_num] = j

        Xtest[num_x*test_num : (num_x+1)*test_num] = tmp_signal[train_num:train_num+test_num]
        Ytest[num_x*test_num : (num_x+1)*test_num] = label_signal
        SNRtest[num_x*test_num : (num_x+1)*test_num] = j

        Xval[num_x*val_num : (num_x+1)*val_num] = tmp_signal[train_num:train_num+val_num]
        Yval[num_x*val_num : (num_x+1)*val_num] = label_signal
        SNRval[num_x*val_num : (num_x+1)*val_num] = j

        num_x = num_x+1
    label_signal = label_signal+1

# index = np.arange(0, NUMBER)
# np.random.shuffle(index)
# # print(index)
DATA_PATH ='/data0/benke/ldx/explainable/dataset'
np.save(DATA_PATH+'radio11CNormTrainX.npy',(Xtrain))
np.save(DATA_PATH+'radio11CNormTrainSnrY',(SNRtrain, Ytrain))
np.save(DATA_PATH+'radio11CNormTestX.npy',(Xtest))
np.save(DATA_PATH+'radio11CNormTestSnrY',(SNRtest, Ytest))
np.save(DATA_PATH+'radio11CValX.npy',(Xval))
np.save(DATA_PATH+'radio11CValSnrY.npy',(SNRval, Yval))
