import csv
import numpy as np
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train','predict'])
parser.add_argument('-model', help="Model name", default= 'RNN_model_weights.h5')
parser.add_argument('-output', help= 'Submission name', default= 'RNN.csv')
args = parser.parse_args()

def WMAE(y_true, y_pred):
	losses = keras.losses.mean_absolute_error(y_true[:, 0], y_pred[:, 0])*200 + keras.losses.mean_absolute_error(y_true[:, 1], y_pred[:, 1])*1 + keras.losses.mean_absolute_error(y_true[:, 2], y_pred[:, 2])*300
	return losses

def train(model_name):
	x_train, y_train, x_valid, y_valid = get_train_data(0.2)
	print(x_train.shape)
	x_train = x_train[:, :5000]
	x_valid = x_valid[:, :5000]
	print(x_train.shape)
	x_train = x_train.reshape(x_train.shape[0], 50, 100)
	x_valid = x_valid.reshape(x_valid.shape[0], 50, 100)
	print(x_train.shape, y_train.shape)
	model = Sequential()
	model.add(LSTM(9, activation = 'relu', input_shape = (50, 100), return_sequences = False))
	model.add(Dense(3, activation = 'tanh'))

	model.output_shape
	model.compile(loss = WMAE, optimizer = 'adam', metrics = ['accuracy'])
	model.fit(x_train, y_train, verbose = 1)

	score = model.evaluate(x_valid, y_valid,verbose=1)
	del x_train, x_valid
	print(score)
	model.save(model_name)
	return

def predict(model_name, file_name):
	x_test = get_test_data()
	x_test = x_test[:, :5000]
	x_test = x_test.reshape(x_test.shape[0], 50, 100)
	model = load_model(model_name, custom_objects={'WMAE': WMAE})
	pred = model.predict(x_test)
	del x_test
	write_submission(pred, file_name)


if __name__ == '__main__':
	if args.mode == 'train':
		print('- TRAIN -')
		train(args.model)
	else:
		print('- PREDICT -')
		predict(args.model, args.output)
