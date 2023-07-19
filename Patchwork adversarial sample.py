import os
import numpy as np
ad_dir = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_VGG16/un_target_attack"
ad_dirs = os.listdir(ad_dir)
ad_save = np.zeros((1, 128, 2))
ad_label = np.zeros(1)
for dir in ad_dirs:
    lab = int(dir[6:])
    dir_path = os.path.join(ad_dir, dir)
    files = os.listdir(dir_path)
    for file in files:
        if "ad" in file:
            ad_npy_path = os.path.join(dir_path, file)
            ad_file = np.load(ad_npy_path)
            label = [lab for i in range(len(ad_file))]
            label = np.array(label)
            ad_save = np.concatenate((ad_save, ad_file), axis=0)
            ad_label = np.concatenate((ad_label, label), axis=0)

print(ad_save[1:].shape)
print(ad_label[1:].shape)
save_dir = "/data0/quziwen/yelinhui/explainable/Deepsig.10A/adversarial_attack/attack_VGG16/对抗样本用于对抗训练/"
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + "ad_example.npy", ad_save)
np.save(save_dir + "source_label.npy", ad_label)