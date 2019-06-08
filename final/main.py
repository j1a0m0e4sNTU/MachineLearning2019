import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from manager import Manager
from model import *
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-bs', help= 'batch size', type= int, default= 64)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-3)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 50)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-csv', help= 'Path to prediction file')
parser.add_argument('-info', help= 'Information to be recorded in file', default= '')
parser.add_argument('-record', help= 'Path to record file')
args = parser.parse_args()

def main():
    model = nn.Linear(200, 3)
    transform = Transform(start=0, end= 200)
    if args.mode == 'train':
        print('Training ...')
        train_set = TrainData('train', transform)
        valid_set = TrainData('valid', transform)
        train_data = DataLoader(dataset= train_set, batch_size= args.bs, shuffle= True)
        valid_data = DataLoader(dataset= valid_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.train(train_data, valid_data)

    else:
        print('Predicting ...')
        test_set = TestData(transform)
        test_data = DataLoader(dataset= test_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.predict(test_data)

if __name__ == '__main__':
    main()
