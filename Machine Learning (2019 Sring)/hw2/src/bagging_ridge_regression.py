import sys
import numpy as np
from matplotlib import pyplot as plt

inverse = lambda vector: np.linalg.inv(vector)

def get_data():
    raw_data = np.genfromtxt('../data/hw2_lssvm_all.dat')
    data_num, feature_num = raw_data.shape
    x = np.ones((data_num, feature_num)).astype(np.float)
    x[:, 1:] = raw_data[:, :-1]
    y = raw_data[:, -1]
    x_train = x[:400]
    x_test  = x[400:]
    y_train = y[:400]
    y_test  = y[400:]
    return x_train, y_train, x_test, y_test 

def ridge_regression(x, y, lambda_value= 0):
    # see page 4 in 206 
    K = x @ (x.T)
    I = np.eye(K.shape[0])
    beta = inverse(lambda_value * I + K) @ y
    weight = (x.T) @ beta
    return weight

def linear_regression(x, y, lambda_value= 0):
    # see page 10 in 09 (ML foundation)
    feature_num = x.shape[1]
    pseudo_inverse = inverse((x.T @ x) + lambda_value * np.eye(feature_num)) @ (x.T)
    weight = pseudo_inverse @ y
    return weight

def evaluate(x, y, weight):
    out = x @ weight
    pred = np.sign(out)
    correct_num = np.sum(pred == y)
    total_num = len(y)
    error_rate = 1 - correct_num / total_num
    return error_rate

def problem_9():
    print('- problem 9 -')
    values = [0.05, 0.5, 5, 50, 500]
    x_train, y_train, x_test, y_test = get_data()
    E_in_array = []
    for value in values:
        weight = ridge_regression(x_train, y_train, lambda_value= value)
        #weight = linear_regression(x_train, y_train, lambda_value= value)
        error = evaluate(x_train, y_train, weight)
        E_in_array.append(error)
    
    print('E_in: {}'.format(E_in_array))
    plt.xlabel('lambda')
    plt.ylabel('E_in rate')
    value_array = [str(value) for value in values]
    plt.plot(value_array, E_in_array)
    plt.savefig('../img/problem_9.png')

def problem_10():
    print('- problem 10 -')
    values = [0.05, 0.5, 5, 50, 500]
    x_train, y_train, x_test, y_test = get_data()
    E_out_array = []
    for value in values:
        weight = ridge_regression(x_train, y_train, lambda_value= value)
        #weight = linear_regression(x_train, y_train, lambda_value= value)
        error = evaluate(x_test, y_test, weight)
        E_out_array.append(error)
    
    print('E_out: {}'.format(E_out_array))
    plt.xlabel('lambda')
    plt.ylabel('E_out rate')
    value_array = [str(value) for value in values]
    plt.plot(value_array, E_out_array)
    plt.savefig('../img/problem_10.png')

def problem_11():
    print('- problem 11 -')
    x_train, y_train, x_test, y_test = get_data()
    sample = np.arange(500)
    values = [0.05, 0.5, 5, 50, 500]
    E_in = []

    for lambda_value in values:
        weight = np.zeros((x_train.shape[1]))
        for _ in range(250):
            sample = [np.random.randint(400) for _ in range(400)]
            x_train_bootstrap = x_train[sample]
            y_train_bootstrap = y_train[sample]
            weight += linear_regression(x_train_bootstrap, y_train_bootstrap, lambda_value)

        E_in.append(evaluate(x_train, y_train, weight))
    
    print('E_in: {}'.format(E_in))
    plt.xlabel('lambda')
    plt.ylabel('E_in rate')
    value_array = [str(value) for value in values]
    plt.plot(value_array, E_in)
    plt.savefig('../img/problem_11.png')

def problem_12():
    print('- problem 12 -')
    x_train, y_train, x_test, y_test = get_data()
    values = [0.05, 0.5, 5, 50, 500]
    E_out = []

    for lambda_value in values:
        weight = np.zeros((x_train.shape[1]))
        for _ in range(250):
            sample = [np.random.randint(400) for _ in range(400)]
            x_train_bootstrap = x_train[sample]
            y_train_bootstrap = y_train[sample]
            weight += linear_regression(x_train_bootstrap, y_train_bootstrap, lambda_value)

        E_out.append(evaluate(x_test, y_test, weight))

    print('E_out: {}'.format(E_out))
    plt.xlabel('lambda')
    plt.ylabel('E_out rate')
    value_array = [str(value) for value in values]
    plt.plot(value_array, E_out)
    plt.savefig('../img/problem_12.png')

def test():
    x = np.eye(8)
    print(x)
    print(inverse(x))

if __name__ == '__main__':
    problem_id = sys.argv[1]
    if problem_id == '9':
        problem_9()
    elif problem_id == '10':
        problem_10()
    elif problem_id == '11':
        problem_11()
    elif problem_id == '12':
        problem_12()
    elif problem_id == 'test':
        test()
    else:
        print('* Please enter 9~12 *')