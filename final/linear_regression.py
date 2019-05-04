import  numpy as np
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train','predict'])
parser.add_argument('-regular', help="Regularization weight", type= float, default= 0)
parser.add_argument('-output', help= 'Submission name', default= 'linear.csv')
args = parser.parse_args()

inverse = lambda vector: np.linalg.inv(vector)

def linear_regression(x, y, lambda_value= 0):
    # see page 10 in 09 (ML foundation)
    feature_num = x.shape[1]
    pseudo_inverse = inverse((x.T @ x) + lambda_value * np.eye(feature_num)) @ (x.T)
    weight = pseudo_inverse @ y
    return weight

def train(regularization):
    x_train, y_train, x_valid, y_valid = get_train_data(0.2)
    pred_train = np.zeros_like(y_train)
    pred_valid = np.zeros_like(y_valid)

    for i in range(3):
        weight = linear_regression(x_train, y_train[:, i], regularization)
        pred_train[:, i] = x_train @ weight
        pred_valid[:, i] = x_valid @ weight
    
    del x_train, x_valid
    print('Train mse: {}'.format(average_mse(pred_train, y_train)))
    print('Valid mse: {}'.format(average_mse(pred_valid, y_valid)))

def predict(file_name, regularization):
    x_train, y_train, _, _ = get_train_data(0)
    x_test = get_test_data()
    pred = np.zeros(x_test.shape[0], 3)
    for i in range(3):
        weight = linear_regression(x_train, y_train, regularization)
        pred[:, i] = x_test @ weight
    
    del x_train
    print('Train mse: {}'.format(average_mse(pred, y_train)))
    write_submission(pred, file_name)

if __name__ == '__main__':
    if args.mode == 'train':
        print('- TRAIN -')
        train(args.regular)
    else:
        print('- PREDICT -')
        predict(args.output, args.regular)