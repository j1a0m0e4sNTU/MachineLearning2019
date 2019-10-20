import numpy as np
from matplotlib import pyplot as plt
import sys

def get_train_data():
    raw_data = np.genfromtxt('train.dat')
    train_x, train_y = raw_data[:, :-1], raw_data[:, -1]
    return train_x, train_y

def get_test_data():
    raw_data = np.genfromtxt('test.dat')
    test_x, test_y = raw_data[:, :-1], raw_data[:, -1]
    return test_x, test_y

def get_distance_matrix(train_data, test_data):
    # data shape: (data_num, feature_dim)
    inner = test_data @ (train_data.T) # (test_size, train_size)
    train_squre = np.diag(train_data @ (train_data.T))
    test_squre = np.diag(test_data @ (test_data.T))
    distance = inner * (-2) + train_squre.reshape(1, -1) + test_squre.reshape(-1, 1)
    return distance

def get_prediction(distance_matrix, y, k= 1):
    # distance_matrix: (test_size, train_size)
    # y : (train_size, )
    test_size = distance_matrix.shape[0]
    score = np.zeros(test_size)
    for i in range(test_size):
        nearest = np.argsort(distance_matrix[i])[:k]
        score[i] = np.sum(y[nearest])
    prediction = np.sign(score)
    return prediction

def get_uniform_prediction(distance_matrix, y, gamma):
    # distance_matrix: (test_size, train_size)
    # y : (train_size, )
    matrix = np.exp((- gamma) * distance_matrix)
    score = matrix @ y
    prediction = np.sign(score)
    return prediction

def get_error_rate(prediction, gt):
    wrong_num = np.sum(prediction != gt)
    return wrong_num / len(gt)

def problem_11():
    k_list = [1, 3, 5, 7, 9]
    train_x, train_y = get_train_data()
    distance = get_distance_matrix(train_x, train_x)
    Ein_list = []
    for k in k_list:
        prediction = get_prediction(distance, train_y, k)
        ein = get_error_rate(prediction, train_y)
        Ein_list.append(ein)
    
    plt.title('Problem 11')
    plt.xlabel('k value')
    plt.ylabel('Ein rate')
    plt.scatter(k_list, Ein_list)
    plt.savefig('img/problem_11.png')

def problem_12():
    k_list = [1, 3, 5, 7, 9]
    train_x, train_y = get_train_data()
    test_x, test_y = get_test_data()
    distance = get_distance_matrix(train_x, test_x)
    Eout_list = []
    for k in k_list:
        prediction = get_prediction(distance, train_y, k)
        eout = get_error_rate(prediction, test_y)
        Eout_list.append(eout)
    
    plt.title('Problem 12')
    plt.xlabel('k value')
    plt.ylabel('Eout rate')
    plt.scatter(k_list, Eout_list)
    plt.savefig('img/problem_12.png')

def problem_13():
    gamma_list = [0.001, 0.1, 1, 10, 100]
    train_x, train_y = get_train_data()
    distance = get_distance_matrix(train_x, train_x)
    Ein_list = []
    for gamma in gamma_list:
        prediction = get_uniform_prediction(distance, train_y, gamma)
        ein = get_error_rate(prediction, train_y)
        Ein_list.append(ein)
    
    plt.title('Problem 13')
    plt.xlabel('gamma')
    plt.ylabel('Ein')
    gamma_list = [str(i) for i in gamma_list]
    plt.scatter(gamma_list, Ein_list)
    plt.savefig('img/problem_13.png')

def problem_14():
    gamma_list = [0.001, 0.1, 1, 10, 100]
    train_x, train_y = get_train_data()
    test_x, test_y = get_test_data()
    distance = get_distance_matrix(train_x, test_x)
    Eout_list = []
    for gamma in gamma_list:
        prediction = get_uniform_prediction(distance, train_y, gamma)
        eout = get_error_rate(prediction, test_y)
        Eout_list.append(eout)
    
    plt.title('Problem 14')
    plt.xlabel('gamma')
    plt.ylabel('Eout')
    gamma_list = [str(i) for i in gamma_list]
    plt.scatter(gamma_list, Eout_list)
    plt.savefig('img/problem_14.png')

def test():
    train_x, train_y = get_train_data()
    print(train_x.shape[0])
    test_x, test_y = get_test_data()
    print(test_x.shape[0])


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        test()
    elif mode == '11':
        problem_11()
    elif mode == '12':
        problem_12()
    elif mode == '13':
        problem_13()
    elif mode == '14':
        problem_14()
    else:
        print('Error: Wrong argument !')