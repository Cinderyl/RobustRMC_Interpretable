from keras.layers.core import Lambda
import matplotlib.pyplot as plt
plt.ion()
import tflearn
import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, MaxPool2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape, Dropout
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import keras.backend as K
import cv2
# K.set_image_data_format('channels_last')
# K.set_learning_phase(0)
import foolbox
import os
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 分配使用率
sess = tf.Session(config=config)
KTF.set_session(sess)

in_shp = list([128, 2])
data_format = 'channels_last'
model = Sequential()
model.add(Reshape((in_shp + [1]), input_shape=in_shp))
model.add(Conv2D(32, (3, 2), activation='relu', data_format=data_format, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1, data_format=data_format, padding='same'))
model.add(Conv2D(64, (3, 2), activation='relu', data_format=data_format, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format, padding='same'))
model.add(Conv2D(128, (3, 2), activation='relu', data_format=data_format, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format, padding='same'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(11))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

filepath = '/data0/quziwen/yelinhui/explainable/Deepsig.10A/alexnet/mytest/model_128_2/对抗训练_alexnet_128_21.h5'
try:
    model.load_weights(filepath)
    print("load model successfule")
except:
    print("fail to load model")


attack_name = ['CW', 'FGSM', 'LB-FGSM', 'JSMA', 'IBM', 'MI-FGSM', 'DeepFool',
               'PGD', 'DeepFoolL2',  # 0-8
               'DeepFoolLinf', 'AdditiveGaussian', 'SaltAndPepper', 'Boundary',
               'NewtonFool', 'RandomPGD']  # 9-14

# DATA_PATH = '/home/yelinhui/yelinhui/explainable/dataset/'
# testX = np.load(DATA_PATH + 'radio11CNormTestX.npy')    # 44000 128 2
# testSnrY = np.load(DATA_PATH + 'radio11CNormTestSnrY.npy') # 2 44000
from foolbox.criteria import TargetClass



'''
目标攻击
'''
# label_to_attack = [0, 1, 4, 5, 7, 8]
# target_label = {1: 7, 8:1, 7:4, 4:7, 0:5, 5:0}
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
#         ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1)
#
#         if np.sum(ad_siganl) != None:
#             ad_siganl_save.append(ad_siganl)    # 攻击成功的样本
#             count = count+1                     # 计算攻击成功的样本的个数
#             count_index.append(index-1)
#             print("success")
#         else:
#             print(index, "failed")
#             print("-------------------")
#         # if count == 2:
#         #     break
#
#     if len(ad_siganl_save) == 0:
#         continue
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#     path = "/home/yelinhui/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/target_attack/label_{}_to_{}/".format(c, target_label[c])
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
# for c in range(11):
#     index = []
#     a = np.array([18, c], dtype=np.float64)      # 需要测试的信号的信噪比和索引
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
#     # import sklearn    # 测试选出的样本是否都是正确的
#     # test_acc = sklearn.metrics.accuracy_score(right_y, right)
#     # print(test_acc)
#
#
#     # 单个样本进行攻击
#     # preprocessing = (np.array([0, 0, 0]), 1)
#     # fmodel = foolbox.models.KerasModel(model, bounds=(-1.5, 1.5),  predicts='probabilities')
#     # attack = foolbox.attacks.FGSM(fmodel, criterion=TargetClass(0))    # 目标攻击
#     # # attack = foolbox.attacks.FGSM(fmodel)                            # 无目标攻击
#     # ad_siganl = attack(right_train[0], label=y_test[0])
#     #
#     # if ad_siganl != None:
#     #     result_ad = model.predict(ad_siganl.reshape(-1, 16, 16))
#     #     print(result_ad)
#     #     print(np.argmax(result_ad))
#     # else:
#     #     print("攻击失败了")
#     # # 多个样本进行攻击
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
#         ad_siganl = attack(example, label=right_y[0], max_epsilon=0.1)
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
#         # if count == 2:
#         #     break
#
#     if len(ad_siganl_save) == 0:
#         continue
#
#     ad_siganl_save = np.array(ad_siganl_save)
#     count_index = np.array(count_index)
#     source_signal = right_train[count_index]
#     path = "/home/yelinhui/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/un_target_attack/label_{}/".format(c)
#     os.makedirs(path, exist_ok=True)
#
#     ad_to_save_path = path + "label_{}_ad_siganl_save.npy".format(c)
#     source_to_save_path = path + "label_{}_source_signal.npy".format(c)
#
#     np.save(ad_to_save_path, ad_siganl_save)
#     np.save(source_to_save_path, source_signal)
#
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


import keras.backend as K
import tensorflow as tf
import cv2
import glob
from os.path import join

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

'''
计算var_list对tensor的梯度
'''
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

def conv_out(model, layer_name):
    for l in model.layers:
        if l.name == layer_name:
            return l.output

num_classes = 11
def grad_cam(input_model, img, category_index, layer_name):
    global conv_output
    ori_img = img
    nb_classes = num_classes
    def target_layer(x): return target_category_loss(  # 好像是定义的交叉熵
        x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    # model.summary()
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
        cam += w * output[:, :, i]    # [32,1]

    _, height, width = ori_img.shape  # 128, 2

    height = 32
    width = 32

    # height = 1080
    # width = 360

    # cam = cam.reshape(16,1)
    # cam = cv2.resize(cam, (16, 16))
    # cam = cam.reshape(128, 2)

    cam_weight = cv2.resize(cam, (1, 128))

    cam = cv2.resize(cam, (width, height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam)
    cam = 255 * cam / np.max(cam)
    print("success")
    return np.uint8(cam), cam_weight


'''
循环画目标攻击的原样本波形图和对抗样本波形图
'''
# dirs_path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/target_attack/"
# dirs = os.listdir(dirs_path)
# for dir in dirs:
#     dir_path = dirs_path+dir
#     files = glob.glob(join(dir_path, f'*.npy'))
#     for file in files:
#         file_split = file.split("_")
#         if file_split[-4] == "ad":
#             ad_path = file
#         else:
#             source_path = file
#
#     source_label = dir.split("_")[1]
#     ad_label = dir.split("_")[-1]
#
#     source_file = np.load(source_path)
#     ad_file = np.load(ad_path)
#
#     path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/target_v/"
#     os.makedirs(path, exist_ok=True)
#
#     path_save_source_dir_xingzuo = path + dir + "/" + "source/xingzuo"    # 保存可视化的路径
#     os.makedirs(path_save_source_dir_xingzuo, exist_ok=True)
#
#     path_save_source_dir_boxing = path + dir + "/" + "source/boxing"  # 保存可视化的路径
#     os.makedirs(path_save_source_dir_boxing, exist_ok=True)
#
#     path_save_ad_dir_xingzuo = path + dir + "/" + "ad/xingzuo"    # 保存可视化的路径
#     os.makedirs(path_save_ad_dir_xingzuo, exist_ok=True)
#
#     path_save_ad_dir_boxing = path + dir + "/" + "ad/boxing"    # 保存可视化的路径
#     os.makedirs(path_save_ad_dir_boxing, exist_ok=True)
#
#     count = 0
#     for img in source_file:
#         img = img.reshape(1, 128, 2)
#         cam, cam_wight = grad_cam(model, img, int(source_label), 'conv2d_3')
#         radio11CamweightTestX = cam_wight
#
#         '''
#         权重归一化
#         '''
#         # norm_cam_weight = cam_wight
#         # if min(norm_cam_weight) < 0:
#         #     norm_cam_weight = norm_cam_weight-min(norm_cam_weight)
#         #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
#         # if min(norm_cam_weight) > 0:
#         #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
#         #
#         # for i in range(128):
#         #     if norm_cam_weight[i] < 0.9:
#         #         norm_cam_weight[i] = 0
#
#         '''
#         画波形权重图
#         '''
#         plt.ion()
#         plt.figure(figsize=(13, 7))
#         e = img.reshape(128, 2)
#         first_a = e[0:128, 0]
#         last_a = e[0:128, 1]
#         plt.scatter(np.arange(128), first_a, cmap = plt.cm.jet, c = radio11CamweightTestX.reshape(128), s=100)
#         plt.scatter(np.arange(128), last_a, cmap = plt.cm.jet, c = radio11CamweightTestX.reshape(128), s=100)
#         # plt.colorbar()
#         # plt.title('Modulation Type:' + str(category[right_y[0]]) + '   Likelihood of '+str(category[right_y[0]]) + ': ' + str(str_possibilities),fontsize='xx-large')
#         outputfile_path = path_save_source_dir_boxing + "/" + "label" + source_label + "波形" + str(
#             count) + ".png"
#         plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
#         # plt.show()
#         plt.close()
#         '''
#         正确的星座图
#         '''
#         plt.figure(figsize=(7, 7))
#         plt.xlim(-1.2, 1.2)
#         plt.ylim(-1.2, 1.2)
#         plt.grid(linestyle='-.')
#         signal = img.reshape(128, 2)
#         first_a = signal[0:128, 0]
#         last_a = signal[0:128, 1]
#         plt.scatter(first_a, last_a, cmap=plt.cm.jet,
#                     c=radio11CamweightTestX.reshape(128), s=100)
#         # plt.colorbar()
#         # plt.title('True label:' + str(category[right_y[0]]) + '   likelihood of label ' +
#         #           str(category[right_y[0]]) + ': ' + str(str_possibilities))
#         outputfile_path = path_save_source_dir_xingzuo + "/" + "label" + source_label + "星座" + str(
#             count) + ".png"
#         count = count+1
#         plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
#         # plt.show()
#         plt.close()
#
#     count = 0
#     for img in ad_file:
#         img = img.reshape(1, 128, 2)
#         '''
#         计算权重信息
#         信号的幅值
#         信号的相位
#         需要注意的是相位和幅值的权重是一样的
#         '''
#         cam, cam_wight = grad_cam(model, img, int(ad_label), 'conv2d_3')
#         radio11CamweightTestX = cam_wight
#
#         '''
#         权重归一化
#         '''
#         # norm_cam_weight = cam_wight
#         # if min(norm_cam_weight) < 0:
#         #     norm_cam_weight = norm_cam_weight-min(norm_cam_weight)
#         #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
#         # if min(norm_cam_weight) > 0:
#         #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
#         #
#         # for i in range(128):
#         #     if norm_cam_weight[i] < 0.9:
#         #         norm_cam_weight[i] = 0
#
#         '''
#         画波形权重图
#         '''
#         plt.ion()
#         plt.figure(figsize=(13, 7))
#         e = img.reshape(128, 2)
#         first_a = e[0:128, 0]
#         last_a = e[0:128, 1]
#         plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=radio11CamweightTestX.reshape(128), s=100)
#         plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=radio11CamweightTestX.reshape(128), s=100)
#         # plt.colorbar()
#         # plt.title('Modulation Type:' + str(category[right_y[0]]) + '   Likelihood of '+str(category[right_y[0]]) + ': ' + str(str_possibilities),fontsize='xx-large')
#         outputfile_path = path_save_ad_dir_boxing + "/" + "label" + source_label + "波形" + str(
#             count) + ".png"
#         plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
#         # plt.show()
#         plt.close()
#
#         '''
#         正确的星座图
#         '''
#         plt.figure(figsize=(7, 7))
#         plt.xlim(-1.2, 1.2)
#         plt.ylim(-1.2, 1.2)
#         plt.grid(linestyle='-.')
#         signal = img.reshape(128, 2)
#         first_a = signal[0:128, 0]
#         last_a = signal[0:128, 1]
#         plt.scatter(first_a, last_a, cmap=plt.cm.jet,
#                     c=radio11CamweightTestX.reshape(128), s=100)
#         # plt.colorbar()
#         # plt.title('True label:' + str(category[right_y[0]]) + '   likelihood of label ' +
#         #           str(category[right_y[0]]) + ': ' + str(str_possibilities))
#         outputfile_path = path_save_ad_dir_xingzuo + "/" + "label" + source_label + "星座" + str(
#             count) + ".png"
#         count = count + 1
#         plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
#         # plt.show()
#         plt.close()


'''
循环画无目标攻击的原样本波形图和对抗样本波形图（或者对抗训练后的波形图）
'''
dirs_path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/un_target_attack/"
dirs = os.listdir(dirs_path)
for dir in dirs:
    dir_path = dirs_path+dir
    files = glob.glob(join(dir_path, f'*.npy'))
    txt_file = glob.glob(join(dir_path, f'*.txt'))[0]
    source_label = dir.split("_")[-1]
    for file in files:
        file_split = file.split("_")
        if file_split[-3] == "ad":
            ad_path = file
        else:
            source_path = file

    source_file = np.load(source_path)
    ad_file = np.load(ad_path)

    path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/对抗训练后的可视化图/"
    os.makedirs(path, exist_ok=True)

    path_save_source_dir_xingzuo = path + dir + "/" + "source/xingzuo"    # 保存可视化的路径
    os.makedirs(path_save_source_dir_xingzuo, exist_ok=True)

    path_save_source_dir_boxing = path + dir + "/" + "source/boxing"  # 保存可视化的路径
    os.makedirs(path_save_source_dir_boxing, exist_ok=True)

    path_save_ad_dir_xingzuo = path + dir + "/" + "ad/xingzuo"    # 保存可视化的路径
    os.makedirs(path_save_ad_dir_xingzuo, exist_ok=True)

    path_save_ad_dir_boxing = path + dir + "/" + "ad/boxing"    # 保存可视化的路径
    os.makedirs(path_save_ad_dir_boxing, exist_ok=True)

    # count = 0
    # for img in source_file:
    #     img = img.reshape(1, 128, 2)
    #     cam, cam_wight = grad_cam(model, img, int(source_label), 'conv2d_3')
    #     radio11CamweightTestX = cam_wight
    #
    #     '''
    #     权重归一化
    #     '''
    #     # norm_cam_weight = cam_wight
    #     # if min(norm_cam_weight) < 0:
    #     #     norm_cam_weight = norm_cam_weight-min(norm_cam_weight)
    #     #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
    #     # if min(norm_cam_weight) > 0:
    #     #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
    #     #
    #     # for i in range(128):
    #     #     if norm_cam_weight[i] < 0.9:
    #     #         norm_cam_weight[i] = 0
    #
    #     '''
    #     画波形权重图
    #     '''
    #     plt.ion()
    #     plt.figure(figsize=(13, 7))
    #     e = img.reshape(128, 2)
    #     first_a = e[0:128, 0]
    #     last_a = e[0:128, 1]
    #     plt.scatter(np.arange(128), first_a, cmap = plt.cm.jet, c = radio11CamweightTestX.reshape(128), s=100)
    #     plt.scatter(np.arange(128), last_a, cmap = plt.cm.jet, c = radio11CamweightTestX.reshape(128), s=100)
    #     # plt.colorbar()
    #     outputfile_path = path_save_source_dir_boxing + "/" + "label" + source_label + "波形" + str(
    #         count) + ".png"
    #     plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
    #     # plt.show()
    #     plt.close()
    #
    #     '''
    #     正确的星座图
    #     '''
    #     plt.figure(figsize=(7, 7))
    #     plt.xlim(-1.2, 1.2)
    #     plt.ylim(-1.2, 1.2)
    #     plt.grid(linestyle='-.')
    #     signal = img.reshape(128, 2)
    #     first_a = signal[0:128, 0]
    #     last_a = signal[0:128, 1]
    #     plt.scatter(first_a, last_a, cmap=plt.cm.jet,
    #                 c=radio11CamweightTestX.reshape(128), s=100)
    #     # plt.colorbar()
    #     outputfile_path = path_save_source_dir_xingzuo + "/" + "label" + source_label + "星座" + str(
    #         count) + ".png"
    #     count = count+1
    #     plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
    #     # plt.show()
    #     plt.close()

    count = 0
    txt_obj = open(txt_file, 'r', encoding='UTF-8')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')[1:-1]  # 文本分割
    list_symbol = []

    for index, img in enumerate(ad_file):
        img = img.reshape(1, 128, 2)
        # ad_label = txt_lines[index][-1]
        # print(ad_label)
        cam, cam_wight = grad_cam(model, img, int(source_label), 'conv2d_3')
        radio11CamweightTestX = cam_wight

        '''
        权重归一化
        '''
        # norm_cam_weight = cam_wight
        # if min(norm_cam_weight) < 0:
        #     norm_cam_weight = norm_cam_weight-min(norm_cam_weight)
        #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
        # if min(norm_cam_weight) > 0:
        #     norm_cam_weight = norm_cam_weight/max(norm_cam_weight)
        #
        # for i in range(128):
        #     if norm_cam_weight[i] < 0.9:
        #         norm_cam_weight[i] = 0

        '''
        画波形权重图
        '''
        plt.ion()
        plt.figure(figsize=(13, 7))
        e = img.reshape(128, 2)
        first_a = e[0:128, 0]
        last_a = e[0:128, 1]
        plt.scatter(np.arange(128), first_a, cmap=plt.cm.jet, c=radio11CamweightTestX.reshape(128), s=100)
        plt.scatter(np.arange(128), last_a, cmap=plt.cm.jet, c=radio11CamweightTestX.reshape(128), s=100)
        # plt.colorbar()
        # plt.title('Modulation Type:' + str(category[right_y[0]]) + '   Likelihood of '+str(category[right_y[0]]) + ': ' + str(str_possibilities),fontsize='xx-large')
        outputfile_path = path_save_ad_dir_boxing + "/" + "label" + source_label + "波形" + str(
            count) + ".png"
        plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
        # plt.show()
        plt.close()

        '''
        正确的星座图
        '''
        plt.figure(figsize=(7, 7))
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(linestyle='-.')
        signal = img.reshape(128, 2)
        first_a = signal[0:128, 0]
        last_a = signal[0:128, 1]
        plt.scatter(first_a, last_a, cmap=plt.cm.jet,
                    c=radio11CamweightTestX.reshape(128), s=100)
        # plt.colorbar()
        # plt.title('True label:' + str(category[right_y[0]]) + '   likelihood of label ' +
        #           str(category[right_y[0]]) + ': ' + str(str_possibilities))
        outputfile_path = path_save_ad_dir_xingzuo + "/" + "label" + source_label + "星座" + str(
            count) + ".png"
        count = count + 1
        plt.savefig(outputfile_path, format='png', dpi=500, bbox_inches='tight')
        # plt.show()
        plt.close()


'''
无目标攻击--用于对抗训练
'''
# DATA_PATH = '/data0/quziwen/yelinhui/explainable/dataset/'
# trainX = np.load(DATA_PATH + 'radio11CNormTrainX.npy')         # 44000 128 2
# trainSnrY = np.load(DATA_PATH + 'radio11CNormTrainSnrY.npy')   # 2 44000
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
#     path = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_Alexnet/无目标攻击--用于对抗训练/label_{}/".format(c)
#     os.makedirs(path, exist_ok=True)
#
#     ad_to_save_path = path + "label_{}_ad_siganl_save.npy".format(c)
#     source_to_save_path = path + "label_{}_source_signal.npy".format(c)
#     label_to_save_path = path + "source_{}_label_of_ad.npy".format(c)
#
#     np.save(ad_to_save_path, ad_siganl_save)
#     np.save(source_to_save_path, source_signal)
#     np.save(label_to_save_path, source_label_of_ad)
