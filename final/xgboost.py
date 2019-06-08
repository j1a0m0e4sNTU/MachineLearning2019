import argparse
import numpy as np
import xgboost 
from util import *
def train(params, train_x, train_y, valid_x, valid_y):
    pred_train = np.zeros_like(train_y)
    pred_valid = np.zeros_like(valid_y)
    valid_matrix = xgboost.DMatrix(valid_x)
    train_matrix = xgboost.DMatrix(train_x)

    for i in range(3):
        matrix = xgboost.DMatrix(train_x, train_y[:, i])
        model = xgboost.train(params, matrix)
        pred_train[:, i] = model.predict(train_matrix)
        pred_valid[:, i] = model.predict(valid_matrix)
        
    train_wmae, train_nae = evaluate(pred_train, train_y)
    valid_wmae, valid_nae = evaluate(pred_valid, valid_y)
    print('Training   || WMAE: {} NAE: {}'.format(train_wmae, train_nae))
    print('Validation || WMAE: {} NAE: {}'.format(valid_wmae, valid_nae))
    
def main():
    train_x, train_y, valid_x, valid_y = get_train_data(0.2)
    train_x, valid_x = train_x[:, :200], valid_x[:, :200]

    params = {'objective':'reg:squarederror'}
    train(params, train_x, train_y, valid_x, valid_y)

if __name__ == '__main__':
    main()