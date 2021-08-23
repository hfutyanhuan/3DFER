import os
import random
import shutil
path = r'D:\Bosphorus_map_dataset'

# for i in os.listdir(path):
#     map_path = os.path.join(path,i)
#     for j in os.listdir(map_path):
#         sub_path = os.path.join(map_path,j)
#         for k in os.listdir(sub_path):
#             filepath = os.path.join(sub_path,k)
#             if k.split('.')[1] == 'jpg':
#                 name = k.split('.')[0] + '.png'
#                 filepath1 = os.path.join(sub_path, name)
#                 os.rename(filepath, filepath1)


###
#切分训练集测试集bosphorus
###
def split_train_test(path, train_path, test_path):
    sub = []
    for i in os.listdir(path):
        sub.append(i)
    random.shuffle(sub)
    for j in range(len(sub)-16):
        sub_path = os.path.join(path, sub[j])
        for k in os.listdir(sub_path):
            ex_path = os.path.join(sub_path, k)
            for m in os.listdir(ex_path):
                train_paths = os.path.join(train_path, sub[j], k)
                if not os.path.exists(train_paths):
                    os.makedirs(train_paths)
                shutil.copy(os.path.join(path, sub[j], k, m), os.path.join(train_paths, m))

    for j in range(64,80):
        sub_path = os.path.join(path, sub[j])
        for k in os.listdir(sub_path):
            ex_path = os.path.join(sub_path, k)
            for m in os.listdir(ex_path):
                test_paths = os.path.join(test_path, sub[j], k)
                if not os.path.exists(test_paths):
                    os.makedirs(test_paths)
                shutil.copy(os.path.join(path, sub[j], k, m), os.path.join(test_paths, m))

if __name__ == '__main__':
    T_path = r'G:\FER_dataset_yan\Bosphorus_map_dataset_60\T'
    #split_train_test()