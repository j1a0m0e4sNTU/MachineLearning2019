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

def random_forest(train_data, tree_num = 3000, bagging= 0.8):
    train_size = train_data.shape[0]
    bagging_size = int(train_size * bagging)
    forest = []
    for i in range(tree_num):
        print('tree: {}/{}'.format(i + 1, tree_num))
        sample_ids = [np.random.randint(train_size) for _ in range(bagging_size)]
        samples = train_data[sample_ids]
        tree = DecisionTree(samples)
        forest.append(tree)

    return forest

def problem_11():
    train_data = get_train_data()
    dt = DecisionTree(train_data, None)
    dt.show()
    print('Height: {}'.format(dt.height()))
    

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
    train_data = get_train_data()
    train_size = train_data.shape[0]
    forest_size = 30000
    forest = random_forest(train_data, forest_size, 0.8)
    error_record = np.zeros((train_size))
    for i, tree in enumerate(forest):
        pred = tree.predict_all(train_data[:, :-1])
        error_count = int(get_error_rate(train_data[:, -1], pred) * train_size)
        error_record[error_count] += 1
    
    plt.title('Problem 14')
    plt.xlabel('Error count (= E_in * 100)')
    plt.ylabel('Tree count')
    x_label = [i for i in range(train_size)]
    plt.bar(x_label, error_record)
    plt.savefig('img/14.png')

def problem_15():
    train_data = get_train_data()
    data_size = train_data.shape[0]
    forest_size = 30000
    forest = random_forest(train_data, forest_size, 0.8)
    predictions = np.zeros((forest_size, data_size))
    E_in = []
    for i, tree in enumerate(forest):
        predictions[i] = tree.predict_all(train_data[:, :-1])
    for i in range(forest_size):
        prediction = np.sign(predictions[: i + 1].sum(0)) 
        prediction[prediction == 0] = 1
        e_in = get_error_rate(train_data[:, -1], prediction)
        E_in.append(e_in)

    plt.title('Problem 15')
    plt.xlabel('Tree num')
    plt.ylabel('E_in rate')
    x_label = [i+1 for i in range(forest_size)]
    plt.plot(x_label, E_in)
    plt.savefig('img/15.png')
    print('Last 10 E_in: {}'.format(E_in[-10:]))

def problem_16():
    train_data = get_train_data()
    test_data  = get_test_data()
    data_size = test_data.shape[0]
    forest_size = 30000
    forest = random_forest(train_data, forest_size, 0.8)
    predictions = np.zeros((forest_size, data_size))
    E_out = []
    for i, tree in enumerate(forest):
        predictions[i] = tree.predict_all(test_data[:, :-1])
    for i in range(forest_size):
        prediction = np.sign(predictions[: i + 1].sum(0)) 
        prediction[prediction == 0] = 1
        e_out = get_error_rate(test_data[:, -1], prediction)
        E_out.append(e_out)
    
    plt.title('Problem 16')
    plt.xlabel('Tree num')
    plt.ylabel('E_out rate')
    x_label = [i+1 for i in range(forest_size)]
    plt.plot(x_label, E_out)
    plt.savefig('img/16.png')
    print('Last 10 E_out: {}'.format(E_out[-10:]))

def test():
    print('-- Test --')
    x = [1, 2, 3]
    y = [1, 4, 9]
    plt.bar(x, y)
    plt.show()

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