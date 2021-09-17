import pickle
import pandas as pd
import numpy as np
import scipy.io as scio
import h5py

def pkl_read(pkl_data_path):
    file = open(pkl_data_path, 'rb')
    database = pickle.load(file)
    file.close()
    return database


def pkl_to_entity_list(pkl_data):
    data_list = []
    for key in pkl_data:
        data_list.append(key)
    return data_list


def txt_read(txt_data_path):
    with open(txt_data_path, 'r') as f:
        my_data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
        # 对list中的数据做分隔和类型转换
        out_data = []
        for line in my_data:
            line_data = line.split()
            out_data.append(line_data)
            numbers_float = map(float, line_data)
    return out_data


def matrix_generating(line, dim, standard):
    initial = -1*np.ones((line, dim))
    for i in range(standard.__len__()):
        for j in range(standard[i].__len__()):
            initial[i][standard[i][j]-1] = 1
    return initial


def matrix_ising(data):
    for i in range(data.__len__()):
        for j in range(data[i].__len__()):
            if data[i][j] == 0:
                data[i][j] = -1


if __name__ == '__main__':
    disease = 'Sumdis'
    date = '(20210904)'
    tokens = pkl_read('tokens/conformity_40_tokens_'+disease+date+'.pkl')  # abstract
    entity_pkl = pkl_read('tokens_list/tokens_list—'+disease+date+'.pkl')  # dict list  
    entity_list = pkl_to_entity_list(entity_pkl)
    print(tokens)

    matrix_result = matrix_generating(int(tokens.__len__()), int(entity_list.__len__()), tokens)
    # print(matrix_result[0].__len__())
    # matrix_result_ising = matrix_ising(matrix_result)
    # np.array(matrix_result)
    print(matrix_result.shape)
    f = h5py.File('Matrix/matrix_ising_conformity_40_'+disease+date+'.h5', 'w')
    f.create_dataset('Ising', data=matrix_result)
    f.close()
    # scio.savemat('8_cancer_data/matrix/matrix_ising_乳腺癌(20210609).mat',  {"Ising":matrix_result})
