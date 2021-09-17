import h5py
import numpy as np
import scipy.io as si

disease = 'HBVR3'
path = 'matlab/MAT/' + disease + '.mat'  # 需要读取的mat文件路径

# demo = si.loadmat(path)
# data = demo['result_use']
feature = h5py.File(path)  # 读取mat文件
print(feature)
data = feature['result_use']
print(data.shape[0])
data_t = np.transpose(data)  # data_t是numpy.ndarray
# 再将其存为npy格式文件
np.save('matlab/NPY/correlation_' + disease + '(20210831).npy', data_t)
