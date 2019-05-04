import numpy as np

x_train_path = '../../data_CSIE_ML/X_train/arr_0.npy'
y_train_path = '../../data_CSIE_ML/Y_train/arr_0.npy'
x_test_path  = '../../data_CSIE_ML/X_test/arr_0.npy'
x_mean_std_path = 'x_mean_std.npy'
y_mean_std_path = 'y_mean_std.npy'
FEATURE_NUM = 3

def normalize(array, mean, std):
    return (array - mean) / std

def get_train_data(validation= 0.2):
    x_train_all = np.load(x_train_path)
    x_mean_std = np.load(x_mean_std_path)
    y_train_all = np.load(y_train_path)
    y_mean_std = np.load(y_mean_std_path)

    x_train_all = normalize(x_train_all, x_mean_std[0], x_mean_std[1])
    y_train_all = normalize(y_train_all, y_mean_std[0], y_mean_std[1])
    
    if 0 < validation < 1:
        cut_size = int((1 - validation) * x_train_all.shape[0])
        x_train, x_valid = x_train_all[:cut_size], x_train_all[cut_size:]
        y_train, y_valid = y_train_all[:cut_size], y_train_all[cut_size:]
        return x_train, y_train, x_valid, y_valid
    else:
        return x_train_all, y_train_all, None, None

def get_test_data():
    x_test_all = np.load(x_test_path)
    x_mean_std = np.load(x_mean_std_path)
    x_test_all = normalize(x_test_all, x_mean_std[0], x_mean_std[1])
    return x_test_all

def average_mse(pred, y):
    num = y.shape[0]
    mse = ((pred - y)**2) / (num * 3)
    return mse