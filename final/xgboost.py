import argparse
import numpy as np
import xgboost 
from util import *

def train(model, train_x, train_y, valid_x, valid_y):
    pred_train = np.zeros_like(train_y)
    pred_valid = np.zeros_like(valid_y)

    for i in range(3):
        model.fit(train_x, train_y)
        pred_train[:, i] = model.predict(train_x)
        pred_valid[:, i] = model.predict(valid_y)
        
    train_wmae, train_nae = evaluate(pred_train, train_y)
    valid_wmae, valid_nae = evaluate(pred_valid, valid_y)
    print('Training   || WMAE: {} NAE: {}'.format(train_wmae, train_nae))
    print('Validation || WMAE: {} NAE: {}'.format(valid_wmae, valid_nae))
    
def main():
    train_x, train_y, valid_x, valid_y = get_sample_data()
    # train_x, valid_x = train_x[:, :200], valid_x[:, :200]

    model = xgboost.sklearn.XGBRegressor()
    train(model, train_x, train_y, valid_x, valid_y)

if __name__ == '__main__':
    main()