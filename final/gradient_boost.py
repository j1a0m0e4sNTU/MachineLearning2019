import  argparse
import numpy as np
from util import *
from loss_manager import LossManager
from linear_regression import linear_regression

def train(steps, regular, x_train, y_train, x_valid, y_valid):
    loss = LossManager()
    alpha_list = []
    hypothesis_list = []
    pred_train = np.zeros_like(y_train)
    pred_valid = np.zeros_like(y_valid)

    for step in range(steps):
        weight_sums = []
        for i in range(3):
            diff = y_train[:, i] - pred_train[:, i]
            weight = linear_regression(x_train, diff, regular)
            pred = (x_train @ weight).reshape((-1, 1))
            alpha  = linear_regression(pred, diff, 0)
            pred_train[:, i] += (x_train @ weight) * alpha
            pred_valid[:, i] += (x_valid @ weight) * alpha
            weight_sums.append(np.sum(weight))
        print(weight_sums)
        loss.evaluate_and_record(step, pred_train, y_train, pred_valid, y_valid)
    
    print('train WMAE: {}'.format(loss.get_train_wmae()))
    print('valid WMAE: {}'.format(loss.get_valid_wmae()))
    title = 'Redularization: {} Step: {}'.format(regular, steps)
    # loss.plot_all(title, 'step', 'gradient-boost-test.png')

def test():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices= ['train', 'predict'])
    parser.add_argument('-steps', help= 'How many hypothesis is to be learned', type= int, default= 10)
    parser.add_argument('-regular', help= 'Regularization weight for linear regression', type= float, default= 1)
    parser.add_argument('-csv', help= 'Prediction csv path')
    args = parser.parse_args()

    if args.mode == 'train':
        x_train, y_train, x_valid, y_valid = get_train_data(0.2)
        x_train = x_train[:, :200]
        x_valid = x_valid[:, :200]
        train(args.steps, args.regular, x_train, y_train, x_valid, y_valid)

    elif args.mode == 'predict':
        pass
