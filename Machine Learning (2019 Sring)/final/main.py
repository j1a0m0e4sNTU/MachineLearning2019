import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from manager import Manager
from model import *
import torch.nn as nn
torch.manual_seed(1004)

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-model', help= 'Model symbol for configuration')
parser.add_argument('-input_dim', help= 'Input dimension', type= int, default= 200)
parser.add_argument('-bs', help= 'batch size', type= int, default= 512)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-3)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 100)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-csv', help= 'Path to prediction file')
parser.add_argument('-info', help= 'Information to be recorded in file', default= '')
parser.add_argument('-record', help= 'Path to record file')
args = parser.parse_args()

def main():
    # model = get_mlp(args.input_dim, args.model)
    model = get_rnn(args.input_dim, args.model, args.bs)
    transform = Transform_04()
    
    if args.mode == 'train':
        print('Training ...')
        train_set = TrainData('train', transform)
        valid_set = TrainData('valid', transform)
        train_data = DataLoader(dataset= train_set, batch_size= args.bs, shuffle= True, drop_last= True)
        valid_data = DataLoader(dataset= valid_set, batch_size= args.bs, drop_last= True)

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
