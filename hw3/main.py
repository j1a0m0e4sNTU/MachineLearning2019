import numpy as np

train_path = 'train.dat'
test_path  = 'test.dat'

def get_data(path):
    array = np.genfromtxt(path)
    return array
    