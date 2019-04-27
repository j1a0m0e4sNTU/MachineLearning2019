import sys
import numpy as np
from matplotlib import pyplot as plt

iteration_number = 300
def get_data():
    data_train = np.genfromtxt('../data/hw2_adaboost_train.dat')
    data_test  = np.genfromtxt('../data/hw2_adaboost_test.dat')
    x_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    x_test  = data_test[:, :-1]
    y_test  = data_test[:, -1]
    return x_train, y_train, x_test, y_test

class DecisionStump():
    def __init__(self, dimension, threashold, pos_neg):
        self.dim = dimension
        self.thr = threashold
        self.pos_neg =  1 if pos_neg >= 0 else -1

    def __call__(self, data):
        select = data[:, self.dim]
        pred = np.sign((select - self.thr) * self.pos_neg)
        return pred

class Adaboost():
    def __init__(self, x_train, y_train, iteration= 300):
        self.iteration_number = iteration
        self.x_train = x_train
        self.y_train = y_train.astype(np.int)
        self.data_num, self.dimension = x_train.shape
        self.weights = np.ones(self.data_num) / self.data_num
        self.stumps = []
        self.alphas = []

        self.indices_each_dim = np.zeros_like(self.x_train).astype(np.int).T
        for dim in range(self.dimension):
            self.indices_each_dim[dim] = self.get_sorted_indices(dim)

    def get_sorted_indices(self, dim):
        pairs = np.empty((self.data_num, 2))
        pairs[:, 0] = np.arange(self.data_num)
        pairs[:, 1] = self.x_train[:, dim]
        
        for i in range(1, self.data_num):
            compare = pairs[i].copy()
            index = i - 1
            while index >= 0:
                if pairs[index, 1] > compare[1]:
                    pairs[index + 1] = pairs[index]
                    pairs[index] = compare
                    index -= 1
                else:
                    break
        indices = pairs[:, 0].astype(np.int)
        return indices

    def train(self):
        for _ in range(self.iteration_number):
            self.add_stump_alpha()
            
    def add_stump_alpha(self):
        best = {
            'dim': None,
            'thr': None,
            'pos_neg': None,
            'wrong': None,
            'error': sys.maxsize
        }
        weights_sum = np.sum(self.weights)
        decision = np.zeros(self.data_num).astype(np.int)
        
        for dim in range(self.dimension):
            permutation = self.indices_each_dim[dim]
            x_train = self.x_train[permutation, dim]
            y_train = self.y_train[permutation]
            weights = self.weights[permutation]
            
            for i in range(self.data_num):
                thr = (x_train[i] - 1) if i == 0 else ((x_train[i -1] + x_train[i]) / 2)
                for value in [-1, 1]:
                    decision[i:] = value
                    decision[:i] = value * (-1)
                    wrong = y_train != decision
                    error = np.sum(weights[wrong]) / weights_sum
                    if error < best['error']:
                        best['dim'] = dim
                        best['thr'] = thr
                        best['pos_neg'] = value
                        best['wrong'] = wrong
                        best['error'] = error

        new_stump = DecisionStump(best['dim'], best['thr'], best['pos_neg'])
        self.stumps.append(new_stump)
        diamond = np.sqrt((1 - best['error']) / best['error'])
        new_alpha = np.log(diamond)
        self.alphas.append(new_alpha)

        wrong_ids = best['wrong']
        permutation = self.indices_each_dim[best['dim']]
        weights = self.weights[permutation]
        weights[wrong_ids == True] *= diamond
        weights[wrong_ids == False] /= diamond
        for i in range(self.data_num):
            self.weights[permutation[i]] = weights[i]
            
    def predict(self, x_test, y_test):
        alphas = np.array(self.alphas)
        data_num = x_test.shape[0]
        prediction = np.empty((data_num, len(alphas)))
        for i, decision_stump in enumerate(self.stumps):
            prediction[:, i] = decision_stump(x_test)
        final_prediction = np.sign(prediction @ alphas)
        accuracy = np.sum(y_test == final_prediction) / data_num
        return accuracy

    def record_Error(self, x_test, y_test):
        Error = []
        for _ in range(self.iteration_number):
            self.add_stump_alpha()
            accuracy = self.predict(x_test, y_test)
            Error.append(1 - accuracy)
        return Error

    def record_U(self):
        U_list = []
        for _ in range(self.iteration_number):
            self.add_stump_alpha()
            U = np.sum(self.weights)
            U_list.append(U)
        return U_list

def test():
    print('- test -')
    iteration = 3
    x_train, y_train, x_test, y_test = get_data()
    adaboost = Adaboost(x_train, y_train, iteration)
    adaboost.train()
    acc = adaboost.predict(x_train, y_train)
    print('accuracy after {} iteration: {}'.format(iteration, acc))

def problem_13():
    print('- Problem 13 -')
    iteration = 300
    x_train, y_train, x_test, y_test = get_data()
    adaboost = Adaboost(x_train, y_train, iteration)
    adaboost.train()
    data_num = x_train.shape[0]
    E_g = []
    for decision_stump in adaboost.stumps:
        prediction = decision_stump(x_train)
        correct = np.sum(prediction == y_train)
        error = 1 - correct / data_num
        E_g.append(error)
    
    print(E_g)
    x_axis = [i for i in range(iteration)]
    plt.xlabel('iteration')
    plt.ylabel('Error rate for each g')
    plt.plot(x_axis, E_g)
    plt.savefig('../img/problem_13.png')
    

def problem_14():
    print('- Problem 14 -')
    iteration = 300
    x_train, y_train, x_test, y_test = get_data()
    adaboost = Adaboost(x_train, y_train, iteration)
    Ein = adaboost.record_Error(x_train, y_train)
    x_axis = [i for i in range(iteration)]
    
    print(Ein)
    plt.xlabel('iteration')
    plt.ylabel('E_in rate')
    plt.plot(x_axis, Ein)
    plt.savefig('../img/problem_14.png')

def problem_15():
    print('- Problem 15 -')
    iteration = 300
    x_train, y_train, x_test, y_test = get_data()
    adaboost = Adaboost(x_train, y_train)
    U_list = adaboost.record_U()

    x_axis = [i for i in range(iteration)]
    print(U_list)
    plt.xlabel('iteration')
    plt.ylabel('U value')
    plt.plot(x_axis, U_list)
    plt.savefig('../img/problem_15.png')

def problem_16():
    print('- Problem 16 -')
    iteration = 300
    x_train, y_train, x_test, y_test = get_data()
    adaboost = Adaboost(x_train, y_train, iteration)
    Eout = adaboost.record_Error(x_test, y_test)
    x_axis = [i for i in range(iteration)]
    
    print(Eout)
    plt.xlabel('iteration')
    plt.ylabel('E_out rate')
    plt.plot(x_axis, Eout)
    plt.savefig('../img/problem_16.png')


if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == '13':
        problem_13()
    elif arg == '14':
        problem_14()
    elif arg == '15':
        problem_15()
    elif arg == '16':
        problem_16()
    elif arg == 'test':
        test()
    else:
        print('Wrong argument !')