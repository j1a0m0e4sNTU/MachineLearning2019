import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util import *

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
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
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
        data = torch.from_numpy(data)
        return data

def test_train():
    print('Test train')
    dataset = TrainData('train')
    dataloader = DataLoader(dataset, batch_size= 8, shuffle= True)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        x ,y = data
        print('Batch {} | {} {}'.format(i, x.size(), y.size()))

def test_test():
    print('test test')
    dataset = TestData()
    dataloader = DataLoader(dataset, batch_size= 8, shuffle= False)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        print('Batch {} | {}'.format(i, data.size()))

if __name__ == '__main__':
    test_train()
    test_test()
