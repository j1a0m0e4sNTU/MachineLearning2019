import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util import *

## Transform
class Transform():
    # Get data from start ~ end 
    def __init__(self, start= 0, end= 200):
        self.start = start
        self.end = end
    
    def __call__(self, data):
        data = data[self.start: self.end]
        return data

class Transform_02():
    # select start ~ end feature, and add quadratic term
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, data):
        length = self.end - self.start
        data_new = np.zeros(2 * length)
        data_select = data[self.start : self.end]
        data_new[:length] = data_select
        data_new[length:] = data_select ** 2
        return  data_new

class Transform_03():
    # select first 200 for RNN inputs
    def __init__(self):
        pass

    def __call__(self, data):
        data_select = data[:200]
        data_select = data_select.reshape(20, 10).copy()
        return data_select

class Transform_04():
    # Add quadratic term for RNN inputs
    def __init__(self):
        pass

    def __call__(self, data):
        data_select = data[:200].reshape(20, 10).copy()
        data_2 = data_select ** 2
        data_new = np.hstack([data_select, data_2])
        return data_new

## Dataset
class TrainData(Dataset):
    def __init__(self, mode= 'train', transform= None):
        super().__init__()
        self.data_x = None
        self.data_y = None
        self.transform = transform

        train_x, train_y, valid_x, valid_y = get_train_data()
        if mode == 'train':
            self.data_x = train_x
            self.data_y = train_y
        else:
            self.data_x = valid_x
            self.data_y = valid_y

    def __len__(self):
        return self.data_y.shape[0]

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        if self.transform:
            x = self.transform(x)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y

class TestData(Dataset):
    def __init__(self, transform= None):
        super().__init__()
        self.data = get_test_data()
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        data = torch.FloatTensor(data)
        return data

def test_train():
    print('Test train')
    transform = Transform(0, 200)
    dataset = TrainData('train', transform)
    dataloader = DataLoader(dataset, batch_size= 8, shuffle= True)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        x ,y = data
        print('Batch {} | {} {}'.format(i, x.size(), y.size()))

def test_test():
    print('test test')
    transform = Transform(0, 200)
    dataset = TestData(transform)
    dataloader = DataLoader(dataset, batch_size= 8, shuffle= False)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        print('Batch {} | {}'.format(i, data.size()))

def test2():
    transform = Transform_04()
    data = np.arange(1000)
    data_new = transform(data)
    print(data_new.shape)

if __name__ == '__main__':
    test2()
