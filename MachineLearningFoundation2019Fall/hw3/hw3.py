import math
import numpy as np
from matplotlib import pyplot as plt

def load_data(path):
    data = np.genfromtxt(path)
    data_x = np.ones(data.shape)
    data_x[:, 1:] = data[:, :-1]
    data_y = data[:, -1]
    return data_x, data_y

def error(x, y, w):
    score = x @ w
    pred = np.ones((len(y)))
    pred[score < 0] = -1
    acc = np.sum(pred == y) / len(y)
    error = 1 - acc
    return error


def gd():
    train_x, train_y = load_data("train.dat")
    test_x, test_y = load_data("test.dat")
    eta = 0.001
    T = 2000
    w = np.zeros((train_x.shape[1]))
    for _ in range(T):
        theta = (-1) * train_y * (train_x @ w)
        theta = 1 / (1 + np.exp(-theta))
        N = train_x.shape[0]
        yx = (-1) * train_x * train_y.reshape(-1, 1)
        grad = (1/N) * (yx.T @ theta)
        w -= eta * grad

    print("E_in : {}".format(error(train_x, train_y, w)))
    print("E_out: {}".format(error(test_x, test_y, w)))

def stochastic_gd():
    train_x, train_y = load_data("train.dat")
    test_x, test_y = load_data("test.dat")
    eta = 0.001
    T = 2000
    w = np.zeros((train_x.shape[1]))
    for t in range(T):
        N = train_x.shape[0]
        idx = t % N
        x = train_x[idx]
        y = train_y[idx]
        theta = (-1) * y * (x @ w)
        theta = 1 / (1 + np.exp(-theta))
        yx = (-1) * x * y
        grad = yx * theta
        w -= eta * grad

    print("E_in : {}".format(error(train_x, train_y, w)))
    print("E_out: {}".format(error(test_x, test_y, w)))

def p78():
    train_x, train_y = load_data("train.dat")
    test_x, test_y = load_data("test.dat")
    eta = 0.01
    T = 2000
    gd_ein = []
    gd_eout = []
    sgd_ein = []
    sgd_eout = []

    w = np.zeros((train_x.shape[1]))
    for _ in range(T):
        theta = (-1) * train_y * (train_x @ w)
        theta = 1 / (1 + np.exp(-theta))
        N = train_x.shape[0]
        yx = (-1) * train_x * train_y.reshape(-1, 1)
        grad = (1/N) * (yx.T @ theta)
        w -= eta * grad
        gd_ein.append(error(train_x, train_y, w))
        gd_eout.append(error(test_x, test_y, w))

    w = np.zeros((train_x.shape[1]))
    for t in range(T):
        N = train_x.shape[0]
        idx = t % N
        x = train_x[idx]
        y = train_y[idx]
        theta = (-1) * y * (x @ w)
        theta = 1 / (1 + np.exp(-theta))
        yx = (-1) * x * y
        grad = yx * theta
        w -= eta * grad
        sgd_ein.append(error(train_x, train_y ,w))
        sgd_eout.append(error(test_x, test_y, w))
    
    plt.title("E in")
    plt.plot(gd_ein, c= "r")
    plt.plot(sgd_ein, c= "b")
    plt.savefig("images/p7.png")
    plt.close()

    plt.title("E out")
    plt.plot(gd_eout, c= "r")
    plt.plot(sgd_eout, c= "b")
    plt.savefig("images/p8.png")
    plt.close()


if __name__ == "__main__":
    p78()