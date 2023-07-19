import xlwt
from keras.layers.core import Lambda
import matplotlib.pyplot as plt
plt.ion()
import cv2
import keras.backend as K
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



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


DATA_PATH = '/home/NewDisk/ldx/explainable/Dataset/Deepsig.test/'


trainX = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/trainX_ad.npy")      # 训练集
trainSnrY = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/train_ad_SnrY.npy")    # 训练集标签

'''
对抗样本测试集
'''
testX = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_ad.npy")
testSnrY = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_ad_SnrY.npy")   # 测试集
testSnr = testSnrY[0]
testY = testSnrY[1]

# DATA_PATH = '/home/NewDisk/liaodnaixn/explainable/Dataset/Deepsig.test/'
# testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')        # 44000 128 2
# testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy')  # 2 44000
# testSnr = testSnrY[0]
# testY = testSnrY[1]

list_val = np.arange(len(trainX)) #list_val==array([0,1,2,,...,155999])   len(testX)==156000
random.shuffle(list_val) #将序列的所有元素随机排序
trainX = trainX[list_val]
trainSnrY = (trainSnrY.T[list_val]).T
trainSnr = trainSnrY[0]
trainY = trainSnrY[1]

list_val = np.arange(len(testX)) #list_val==array([0,1,2,,...,155999])   len(testX)==156000
val_percent = 0.1
valX = testX[list_val[0: int(len(testX) * val_percent)]]
valY = testY[list_val[0: int(len(testX) * val_percent)]]
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))
# (176000, 128, 2) (176000,) (4400, 128, 2) (4400,) (44000, 128, 2) (44000,)

num_classes = 11
trainY = tflearn.data_utils.to_categorical(trainY, num_classes)   # 将索引转换为one-hot编码
valY = tflearn.data_utils.to_categorical(valY, num_classes)
testY = tflearn.data_utils.to_categorical(testY, num_classes)
print(np.shape(trainX), np.shape(trainY), np.shape(valX), np.shape(valY), np.shape(testX), np.shape(testY))
'(176000, 128, 2) (176000, 11) (4400, 128, 2) (4400, 11) (44000, 128, 2) (44000, 11)'

trainX = np.reshape(trainX, (len(trainX), 16, 16));
valX = np.reshape(valX, (len(valX), 16, 16));
testX = np.reshape(testX, (len(testX), 16, 16));
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
epochs = 20  # number of epochs to train on
batch_size = 64  # training batch size default 1024

filepath = '/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/NIN.h5'  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
log_filename = 'model_train.csv'
try:
    model.load_weights(filepath)
    print("加载模型成功!")
except:
    print("加载模型失败!")

ad_files = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/用于画AT后的图的对抗样本和原样本/ad/ad5.npy")
ad_files = ad_files.reshape(len(ad_files), 16, 16)
result = model.predict(ad_files[19:20])
result = np.argmax(result)
print(result)


'''
训练部分
'''
# history = model.fit(trainX,
#     trainY,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=1,
#     validation_data=(valX, valY),
#     callbacks = [ #回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
#        keras.callbacks.CSVLogger(log_filename, separator=',', append=True),
#        keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto'),
#        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
#                                     write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                                     embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
#        # keras.callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='auto')
#     ]) #EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
# model_name = filepath
# model.save(filepath)
# model.save_weights(filepath)


'''
测试模型准确性，并获得混淆矩阵
'''
# y_test = testSnrY[1]
# y_predict = np.zeros(shape=len(testX), dtype=np.int32)    # 44000
# pre_batch = 1000
# iter = int(len(testX) / pre_batch)   # 44
# for i in range(0, iter):
#     y_predict[i * pre_batch: (i + 1) * pre_batch] = np.argmax(model.predict(testX[i * pre_batch: (i + 1) * pre_batch]),
#                                                               axis=1)
# import sklearn
# test_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
# test_CM = sklearn.metrics.confusion_matrix(y_test, y_predict)
# print("test_acc:", test_acc, '\n confusion_matrix:\n', test_CM)
# np.save('/home/NewDisk/ldx/explainable/Deepsig.10A/vgg16/TESTconfusion.npy', (y_predict, y_test, testSnrY[0]))
# np.save('CM_140_正常', test_CM)


'''
测试模型对各个信噪比的准确性
'''
# accuracy = np.zeros(20)
# c = 0
# for snr in range(-20, 20, 2):
#     index = []
#     a = np.array([snr], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引，count其实就是找到的索引，0是snr，1是信号标签
#         if i[0] == a[0]:
#             index.append(count)
#         count = count+1
#     index8 = np.array(index)    # 所找到的索引
#     train = testX[index]
#     train = np.reshape(train, (len(train), 128, 2))   # 测试数据
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#     import sklearn
#     y_test = testSnrY[1]
#     y_test = y_test[index]
#     test_acc = sklearn.metrics.accuracy_score(y_test, result)
#     accuracy[c] = test_acc
#     c = c+1
# np.save("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/模型在各个信噪比下的识别准确率--对抗", accuracy)

'''
计算模型在各个信噪比下，对各个信号类别的识别准确率
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
#
# for c in range(11):
#     colum = colum + 1
#     worksheet.write(0, colum, str(category[c]))
#     row = 0
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
#         import sklearn
#         y_test = testSnrY[1]
#         y_test = y_test[index]
#         test_acc = sklearn.metrics.accuracy_score(y_test, result)
#         worksheet.write(row, colum, test_acc)
#         result_save[row-1, colum-1] = test_acc
#         np.save("/home/NewDisk/ldx/explainable/Deepsig.10A/LSTM特征可视化/LSTM模型在各个类信噪比下对各个类的识别准确率--对抗.npy", result_save)
#         print("对应类别的类标{}_信噪比{}_准确率{}".format(c, snr, test_acc))
#     # workbook.save('./模型在各个信噪比下的识别准确率.xls')

'''
测试信号对某个信噪比的某个类的信号的识别准确率
'''
# index = []
# a = np.array([18, 10], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# result = model.predict(train)
# result = np.argmax(result, axis=1)
# print(result)
# import sklearn
# y_test = testSnrY[1]
# y_test = y_test[index]
# test_acc = sklearn.metrics.accuracy_score(y_test, result)
# print(test_acc)

'''
每一个信噪比的准确性
'''
# index = []
# a = np.array([18, 8], dtype=np.float64)    # 需要测试的信号的信噪比和索引
# count = 0
# for i in testSnrY.T:                         # 找所需要测试的信号的索引
#     if i[0] == a[0] and i[1] == a[1]:
#         index.append(count)
#     count = count+1
# index = np.array(index)    # 所找到的索引
# train = testX[index]
# train = np.reshape(train, (len(train), 128, 2))   # 测试数据
# result = model.predict(train)
# result = np.argmax(result, axis=1)
# import sklearn
# y_test = testSnrY[1]
# y_test = y_test[index]
# test_acc = sklearn.metrics.accuracy_score(y_test, result)
# print(test_acc)

'''
输出模型对某个信噪比下11个类的信号的识别准确率
'''
# for c in range(11):
#     index = []
#     a = np.array([16, c], dtype=np.float64)    # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:                         # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count+1
#     index = np.array(index)    # 所找到的索引
#
#     train = testX[index]
#     train = np.reshape(train, (len(train), 16, 16));   # 测试数据
#     result = model.predict(train)
#     result = np.argmax(result, axis=1)
#     # print(len(result))
#     import sklearn
#     y_test = testSnrY[1]
#     y_test = y_test[index]
#     test_acc = sklearn.metrics.accuracy_score(y_test, result)
#     print(test_acc)

'''
画特征图部分
'''
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

def conv_out(model, layer_name):
    for l in model.layers:
        if l.name == layer_name:
            return l.output

def grad_cam(input_model, img, category_index, layer_name):
    global conv_output
    ori_img = img
    nb_classes = num_classes
    def target_layer(x): return target_category_loss(  # 好像是定义的交叉熵
        x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    loss = K.sum(model.output)
    # 遍历各层，当layer_name=="block5_conv3"时 即得到layer_name对应的输出conv_output，
    # conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    '''
    for l in model.layers:
        #print (l)
        if l.name == layer_name:
            conv_output = l.output
    '''
    conv_output = conv_out(model, layer_name)
    # 获取卷积输出对loss的梯度矩阵,并作normalize
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([img])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    # 每个滤波器的梯度求和并取均值获得各滤波器对应权重
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    _, width, height = ori_img.shape    # 16x16

    height = 540
    width = 540

    # height = 1080
    # width = 360

    # cam = cam.reshape(16,1)
    # cam = cv2.resize(cam, (16, 16))
    # cam = cam.reshape(128, 2)

    cam_weight = cv2.resize(cam, (1, 128))

    # cam = cv2.resize(cam, (1, 128))
    cam = cv2.resize(cam, (width, height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam)
    cam = 255 * cam / np.max(cam)
    print("success")
    return np.uint8(cam), cam_weight


category = {0:"WBFM", 1:"QPSK", 2:"QAM64", 3: "QAM16", 4:"PAM4", 5:"GFSK", 6:"CPFSK" ,7:"BPSK", 8:"8PSK", 9:"AM-SSB", 10:"AM-DSB"}
source_test = np.load("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/test_source.npy")

'''
对抗训练前
'''
# number = 40
# for c in range(11):
#     index = []
#     a = np.array([16, c], dtype=np.float64)  # 需要测试的信号的信噪比和索引
#     count = 0
#     for i in testSnrY.T:  # 找所需要测试的信号的索引
#         if i[0] == a[0] and i[1] == a[1]:
#             index.append(count)
#         count = count + 1
#     index = np.array(index)  # 所找到的索引
#     train = testX[index]     # 符合条件的对抗样本
#     # train = np.reshape(train, (len(train), 16, 16));  # 测试数据
#     # y_test = testSnrY[1]
#     # y_test = y_test[index]
#     # result = model.predict(train)
#     # result = np.argmax(result, axis=1)
#     ad_to_source = source_test[index]     # 对抗样本所对应的原文件
#     ad_to_source = ad_to_source.reshape(len(ad_to_source), 16, 16)
#     result = model.predict(ad_to_source)
#     result = np.argmax(result, axis=1)
#
#
#     print("----processing data----")
#     print("----找出对抗训练后能够被正确预测的对抗样本----")
#     count = 0
#     right_index = []
#     for k in result:
#         if k == a[1]:
#             right_index.append(count)
#         count = count + 1
#
#     if right_index != []:
#         right_index = np.array(right_index)      # 找到的所有预测正确的数据的索引
#         right_train = ad_to_source[right_index]  # 找到对抗训练前能够被正确识别的样本
#
#     else:
#         right_train = []
#         right_y = []  # 找到的所有预测正确的数据的类标
#
#     ad = train[right_index]                   # 找到与ad_to_source对应的对抗样本
#     ad = ad.reshape(len(ad), 16, 16)
#     result_ad = model.predict(ad)
#     result_ad = np.argmax(result_ad, axis=1)
#
#     np.save("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/用于话AT后的图的对抗样本和原样本/source/source{}.npy".format(c), right_train)
#     np.save("/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/用于话AT后的图的对抗样本和原样本/ad/ad{}.npy".format(c), ad)
#
#     basepath = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/图/对抗训练后/"
#     if os.path.exists(basepath) == True:
#         pass
#     else:
#         os.makedirs(basepath)
#
#     output_wave = basepath + "波形图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_wave) == True:
#         pass
#     else:
#         os.makedirs(output_wave)
#
#     output_xingzuo = basepath + "星座图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_xingzuo) == True:
#         pass
#     else:
#         os.makedirs(output_xingzuo)
#
#     output_feature = basepath + "特征图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_feature) == True:
#         pass
#     else:
#         os.makedirs(output_feature)
#
#     count = 0
#     for exam in right_train:
#         exam = exam.reshape(1, 16, 16)
#         output_path_feature = output_feature + "/label" + str(c) + '特征{}'.format(count) + '.png'
#         cam, cam_weight = grad_cam(model, exam, c, 'conv2d_9')
#         cv2.imwrite(output_path_feature, cam)
#
#         '''
#         星座图
#         '''
#         plt.figure(figsize=(7, 7))
#         plt.xlim(-1.2, 1.2)
#         plt.ylim(-1.2, 1.2)
#         plt.grid(linestyle='-.')
#         a = exam.reshape(128, 2)
#         first_a = a[0:128, 0]
#         last_a = a[0:128, 1]
#         plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#         plt.colorbar()
#         output_path_xingzuo = output_xingzuo + "/label" + str(c) + '星座{}'.format(count) + '.png'
#         plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#         plt.close()
#         '''
#         波形图
#         '''
#         plt.ion()
#         plt.figure(figsize=(13, 7))
#         plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c = cam_weight.reshape(128), s=100)
#         plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c = cam_weight.reshape(128), s=100)
#         plt.colorbar()
#         output_path_wave = output_wave + "/label" + str(c) + '波形{}'.format(count) + '.png'
#         plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#         plt.close()
#         count = count + 1
#         if count == number:
#             break
#
#     count = 0
#     i = 0
#     for exam in ad:
#         exam = exam.reshape(1, 16, 16)
#         output_path_feature = output_feature + "/label" + str(c) + '特征{}_ad'.format(count) + '.png'
#         index_ad = result_ad[i]
#         cam, cam_weight = grad_cam(model, exam, index_ad, 'conv2d_9')
#         cv2.imwrite(output_path_feature, cam)
#         i = i+1
#         '''
#         星座图
#         '''
#         plt.figure(figsize=(7, 7))
#         plt.xlim(-1.2, 1.2)
#         plt.ylim(-1.2, 1.2)
#         plt.grid(linestyle='-.')
#         a = exam.reshape(128, 2)
#         first_a = a[0:128, 0]
#         last_a = a[0:128, 1]
#         plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#         plt.colorbar()
#         output_path_xingzuo = output_xingzuo + "/label" + str(c) + '星座{}_ad'.format(count) + '.png'
#         plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#         plt.close()
#         '''
#         波形图
#         '''
#         plt.ion()
#         plt.figure(figsize=(13, 7))
#         plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#         plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#         plt.colorbar()
#         output_path_wave = output_wave + "/label" + str(c) + '波形{}_ad'.format(count) + '.png'
#         plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#         plt.close()
#         count = count + 1
#         if count == number:
#             break



'''
对抗训练后
'''
# path_ad = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/用于话AT后的图的对抗样本和原样本/ad"
# path_source = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/用于话AT后的图的对抗样本和原样本/source"
#
# ad_files = os.listdir(path_ad)
# source_files = os.listdir(path_source)

### 画图部分
# for ad_file in ad_files:
#     ad_file_number = ad_file[2:-4]
#     ad_file_number = int(ad_file_number)
#     basepath = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/重新生成对抗样本/图/对抗训练后/"
#
#     if os.path.exists(basepath) == True:
#         pass
#     else:
#         os.makedirs(basepath)
#
#     output_wave = basepath + "波形图/" + "lable{}".format(ad_file_number) + str(category[ad_file_number])
#     if os.path.exists(output_wave) == True:
#         pass
#     else:
#         os.makedirs(output_wave)
#
#     output_xingzuo = basepath + "星座图/" + "lable{}".format(ad_file_number) + str(category[ad_file_number])
#     if os.path.exists(output_xingzuo) == True:
#         pass
#     else:
#         os.makedirs(output_xingzuo)
#
#     output_feature = basepath + "特征图/" + "lable{}".format(ad_file_number) + str(category[ad_file_number])
#     if os.path.exists(output_feature) == True:
#         pass
#     else:
#         os.makedirs(output_feature)
#
#
#     for source_file in source_files:
#         source_file_number = int(source_file[6:-4])
#         if ad_file_number == source_file_number:
#             adver = np.load(path_ad+"/"+ad_file)
#             source = np.load(path_source+"/"+source_file)
#             for i in range(len(adver)):
#                 exam_ad = adver[i].reshape(1, 16, 16)
#                 exam_source = source[i].reshape(1, 16, 16)
#
#                 output_path_feature_ad = output_feature + "/label" + str(ad_file_number) + '特征{}_ad'.format(i) + '.png'
#                 output_path_feature_source = output_feature + "/label" + str(ad_file_number) + '特征{}_source'.format(i) + '.png'
#
#                 cam_ad, cam_weight_ad = grad_cam(model, exam_ad, ad_file_number, 'conv2d_9')
#                 cam_source, cam_weight_source = grad_cam(model, exam_source, ad_file_number, 'conv2d_9')
#                 cv2.imwrite(output_path_feature_ad, cam_ad)
#                 cv2.imwrite(output_path_feature_source, cam_source)
#
#                 '''
#                 星座图
#                 '''
#                 plt.figure(figsize=(7, 7))
#                 plt.xlim(-1.2, 1.2)
#                 plt.ylim(-1.2, 1.2)
#                 plt.grid(linestyle='-.')
#                 a = exam_ad.reshape(128, 2)
#                 first_a = a[0:128, 0]
#                 last_a = a[0:128, 1]
#                 plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight_ad.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_xingzuo = output_xingzuo + "/label" + str(ad_file_number) + '星座{}_ad'.format(i) + '.png'
#                 plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#                 '''
#                 波形图
#                 '''
#                 plt.ion()
#                 plt.figure(figsize=(13, 7))
#                 plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c = cam_weight_ad.reshape(128), s=100)
#                 plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c = cam_weight_ad.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_wave = output_wave + "/label" + str(ad_file_number) + '波形{}_ad'.format(i) + '.png'
#                 plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#
#
#                 '''
#                 原样本星座图
#                 '''
#                 plt.figure(figsize=(7, 7))
#                 plt.xlim(-1.2, 1.2)
#                 plt.ylim(-1.2, 1.2)
#                 plt.grid(linestyle='-.')
#                 a = exam_source.reshape(128, 2)
#                 first_a = a[0:128, 0]
#                 last_a = a[0:128, 1]
#                 plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight_source.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_xingzuo = output_xingzuo + "/label" + str(ad_file_number) + '星座{}_source'.format(i) + '.png'
#                 plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#                 '''
#                 原样本波形图
#                 '''
#                 plt.ion()
#                 plt.figure(figsize=(13, 7))
#                 plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c = cam_weight_source.reshape(128), s=100)
#                 plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c = cam_weight_source.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_wave = output_wave + "/label" + str(ad_file_number) + '波形{}_source'.format(i) + '.png'
#                 plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#
#                 if i == 39:
#                     break



# path = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/test_ad"
# files = os.listdir(path)
#
# path_source = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/test_source"
# files_sources = os.listdir(path_source)
#
# count_number = 80
# for file in files:
#     number = file[12:-19]
#     c = int(number)
#     basepath = "/home/NewDisk/ldx/explainable/Deepsig.10A/NIN/对抗训练/图/对抗后/"
#
#     if os.path.exists(basepath) == True:
#         pass
#     else:
#         os.makedirs(basepath)
#
#     output_wave = basepath + "波形图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_wave) == True:
#         pass
#     else:
#         os.makedirs(output_wave)
#
#     output_xingzuo = basepath + "星座图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_xingzuo) == True:
#         pass
#     else:
#         os.makedirs(output_xingzuo)
#
#     output_feature = basepath + "特征图/" + "lable{}".format(c) + str(category[c])
#     if os.path.exists(output_feature) == True:
#         pass
#     else:
#         os.makedirs(output_feature)
#
#     count = 0
#     for files_source in files_sources:    # 找到了与对抗样本对应的原样本
#         ad = np.load(path+"/"+file)
#         source = np.load(path_source+"/"+files_source)
#         for i in range(len(ad)):
#             if np.sum(ad[i]-source[i])!=0:
#                 example_ad = ad[i].reshape(1,16,16)
#                 output_path_feature = output_feature + "/label" + str(number) + '特征{}_ad'.format(count) + '.png'
#                 cam, cam_weight = grad_cam(model, example_ad, c, 'conv2d_9')
#                 cv2.imwrite(output_path_feature, cam)
#
#                 '''
#                 星座图
#                 '''
#                 plt.figure(figsize=(7, 7))
#                 plt.xlim(-1.2, 1.2)
#                 plt.ylim(-1.2, 1.2)
#                 plt.grid(linestyle='-.')
#                 a = example_ad.reshape(128, 2)
#                 first_a = a[0:128, 0]
#                 last_a = a[0:128, 1]
#                 plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_xingzuo = output_xingzuo + "/label" + str(c) + '星座{}_ad'.format(count) + '.png'
#                 plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#                 '''
#                 波形图
#                 '''
#                 plt.ion()
#                 plt.figure(figsize=(13, 7))
#                 plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c = cam_weight.reshape(128), s=100)
#                 plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c = cam_weight.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_wave = output_wave + "/label" + str(c) + '波形{}_ad'.format(count) + '.png'
#                 plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#
#
#                 '''
#                 画原信号的
#                 '''
#                 example_source = source[i].reshape(1, 16, 16)
#                 output_path_feature = output_feature + "/label" + str(c) + '特征{}_source'.format(count) + '.png'
#                 cam, cam_weight = grad_cam(model, example_source, c, 'conv2d_9')
#                 cv2.imwrite(output_path_feature, cam)
#
#                 '''
#                 星座图
#                 '''
#                 plt.figure(figsize=(7, 7))
#                 plt.xlim(-1.2, 1.2)
#                 plt.ylim(-1.2, 1.2)
#                 plt.grid(linestyle='-.')
#                 a = example_ad.reshape(128, 2)
#                 first_a = a[0:128, 0]
#                 last_a = a[0:128, 1]
#                 plt.scatter(first_a, last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_xingzuo = output_xingzuo + "/label" + str(c) + '星座{}_source'.format(count) + '.png'
#                 plt.savefig(output_path_xingzuo, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#                 '''
#                 波形图
#                 '''
#                 plt.ion()
#                 plt.figure(figsize=(13, 7))
#                 plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#                 plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=cam_weight.reshape(128), s=100)
#                 plt.colorbar()
#                 output_path_wave = output_wave + "/label" + str(c) + '波形{}_source'.format(count) + '.png'
#                 plt.savefig(output_path_wave, format='png', dpi=500, bbox_inches='tight')
#                 plt.close()
#                 count = count + 1
#                 if count == count_number:
#                     break
#             if count == count_number:
#                 break
#         if count == count_number:
#             break
