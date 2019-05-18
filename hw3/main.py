import sys
import numpy as np
from matplotlib import pyplot as plt
from decision_tree import DecisionTree

def get_train_data():
    data = np.genfromtxt('train.dat')
    return data

def get_test_data():
    data = np.genfromtxt('test.dat')
    return data

def get_error_rate(ground_truth, prediction):
    error_count = np.sum(ground_truth != prediction)
    return error_count / len(ground_truth)

def problem_11():
    train_data = get_train_data()
    dt = DecisionTree(train_data, None)
    print('Height: {}'.format(dt.height()))
    dt.root.show_info()
    dt.root.left_node.show_info()
    dt.root.right_node.show_info()

def problem_12():
    train_data = get_train_data()
    test_data = get_test_data()
    dt = DecisionTree(train_data, None)
    E_in = get_error_rate(train_data[:, -1], dt.predict_all(train_data[:, :-1]))
    E_out = get_error_rate(test_data[:, -1], dt.predict_all(test_data[:, :-1]))
    print('E in: {}'.format(E_in))
    print('E out: {}'.format(E_out))

def problem_13():
    train_data = get_train_data()
    test_data  = get_test_data()
    height = []
    E_in = []
    E_out = []

    full_height = 5
    for h in range(1, full_height + 1):
        dt = DecisionTree(train_data, h)
        e_in = get_error_rate(train_data[:, -1], dt.predict_all(train_data[:, :-1]))
        e_out = get_error_rate(test_data[:, -1], dt.predict_all(test_data[:, :-1]))
        height.append(h)
        E_in.append(e_in)
        E_out.append(e_out)
    
    plt.title('Red: Ein | Blue: Eout')
    plt.xlabel('Height of Decision Tree')
    plt.ylabel('Error rate')
    plt.plot(height, E_in, c= 'r')
    plt.plot(height, E_out, c= 'b')
    plt.savefig('img/13.png')

def problem_14():
    pass

def problem_15():
    pass

def problem_16():
    pass

def test():
    print('-- Test --')
    data = get_train_data()
    dt = DecisionTree(data, None)

    error_count = get_error_rate(data[:, -1] ,dt.predict_all(data[:, :-1]))
    print('Height: ', dt.height())
    print("Error count:", error_count)

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == 'test':
        test()    
    elif arg == '11':
        problem_11()
    elif arg == '12':
        problem_12()
    elif arg == '13':
        problem_13()
    elif arg == '14':
        problem_14()
    elif arg == '15':
        problem_15()
    elif arg == '16':
        problem_16()
    else:
        print('Wrong argument !')