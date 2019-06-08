import  numpy as np
import argparse
from util import *
from loss_manager import LossManager


inverse = lambda vector: np.linalg.inv(vector)

def linear_regression(x, y, lambda_value= 0):
    # see page 10 in 09 (ML foundation)
    feature_num = x.shape[1]
    pseudo_inverse = inverse((x.T @ x) + lambda_value * np.eye(feature_num)) @ (x.T)
    weight = pseudo_inverse @ y
    return weight

def train(x_train, y_train, x_valid, y_valid, regularization):
    pred_train = np.zeros_like(y_train)
    pred_valid = np.zeros_like(y_valid)

    for i in range(3):
        weight = linear_regression(x_train, y_train[:, i], regularization)
        pred_train[:, i] = x_train @ weight
        pred_valid[:, i] = x_valid @ weight
    
    del x_train, x_valid
    
    train_wmae, train_nae = evaluate(pred_train, y_train)
    valid_wmae, valid_nae = evaluate(pred_valid, y_valid)
    print('== Regularization: {} =='.format(regularization))
    print('     |   WMAE_error   |   NAE_error')
    print('train|    {:9f}   |   {:9f}'.format(train_wmae, train_nae))
    print('valid|    {:9f}   |   {:9f}'.format(valid_wmae, valid_nae))
    return train_wmae, train_nae, valid_wmae, valid_nae

def predict(file_name, regularization):
    x_train, y_train, _, _ = get_train_data(0)
    x_test = get_test_data()
    pred = np.zeros((x_test.shape[0], 3))
    for i in range(3):
        weight = linear_regression(x_train, y_train[:, i], regularization)
        pred[:, i] = x_test @ weight
    
    del x_train, x_test
    write_submission(pred, file_name)

def expiriment(regular):
    x_train, y_train, x_valid, y_valid = get_train_data(0.2)
    x_train_new, x_valid_new = x_train[:, :200], x_valid[:, :200]
    #x_train_new, x_valid_new = np.ones((x_train.shape[0], 201)), np.ones((x_valid.shape[0], 201))
    #x_train_new[:, :200], x_valid_new[:, :200] = x_train[:, :200], x_valid[:, :200]
    train(x_train_new, y_train, x_valid_new, y_valid, regular)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','predict', 'exp'])
    parser.add_argument('-regular', help="Regularization weight", type= float, default= 0)
    parser.add_argument('-output', help= 'Submission name', default= 'linear.csv')
    parser.add_argument('-record', help= 'REcord file name', default= 'record.txt')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print('- TRAIN -')
        x_train, y_train, x_valid, y_valid = get_train_data(0.2)
        train(x_train, y_train, x_valid, y_valid, args.regular)
    
    elif args.mode == 'predict':
        print('- PREDICT -')
        predict(args.output, args.regular)
    
    elif args.mode == 'exp':
        print('- Experiment -')
        expiriment(args.regular)
