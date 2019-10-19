import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse

x_train_path = '../../data_CSIE_ML/X_train/arr_0.npy'
y_train_path = '../../data_CSIE_ML/Y_train/arr_0.npy'
x_test_path  = '../../data_CSIE_ML/X_test/arr_0.npy'

def save_mean_std():
    data_train = np.load(x_train_path)
    data_test  = np.load(x_test_path)
    data = np.vstack([data_train, data_test])
    del data_train
    del data_test
    print(data.shape)
    mean_std = np.zeros((2, data.shape[1]))
    mean_std[0] = np.mean(data, 0)
    mean_std[1] = np.std(data, 0)
    print(mean_std.shape)
    np.save('mean_std.npy', mean_std)


def plot_figure(txt_path, name, mode= 'loss'):    
    figure_dir = 'records/deeplearning'
    file = open(txt_path, 'r')
    lines = []
    for line in file:
        lines.append(line)
    
    train_data = []
    valid_data = []
    stride = len(mode) + 2
    for i in range(len(lines)):
        if lines[i][0] == '-':
            trian_tokens = lines[i + 1].split(' ')
            valid_tokens = lines[i + 2].split(' ')
            train_d, valid_d = None, None
            if mode.upper() == 'LOSS':
                train_d = float(trian_tokens[3])
                valid_d = float(valid_tokens[3])
            elif mode.upper() == 'WMAE':
                train_d = float(trian_tokens[6])
                valid_d = float(valid_tokens[6])
            elif mode.upper() == 'NAE':
                train_d = float(trian_tokens[9])
                valid_d = float(valid_tokens[9])
            train_data.append(train_d)
            valid_data.append(valid_d)
    
    x_axis = [i for i in range(len(train_data))]
    plt.plot(x_axis, train_data, c= 'b')
    plt.plot(x_axis, valid_data, c= 'r')
    plt.title(mode.upper())
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(figure_dir, name + '_' + mode + '.png'))
    plt.close()

def test():
    print('test')
    path = sys.argv[1]
    plot_figure(path)

def main():
    path = sys.argv[1]
    name = path.split('/')[-1][:-4]
    plot_figure(path, name, 'loss')
    plot_figure(path, name, 'wmae')
    plot_figure(path, name, 'nae')

if __name__ == '__main__':
    main()