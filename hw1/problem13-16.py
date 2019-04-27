import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from qpsolvers import solve_qp
import sys
sys.path.append('libsvm-3.23/python/')
from svmutil import *
import scipy
from numpy import random as random

train_data_path = 'features.train'
test_data_path = 'features.test'

def read_data(path):
    data_raw = np.genfromtxt(path)
    y = data_raw[:, 0]
    x = data_raw[:, 1:]
    return y, x

def sv_to_numpy(sv):
    '''Trasform support vector from dict to numpy'''
    i = 1
    sv_new = []
    while True:
        sv_new.append(sv[i])
        i += 1
        if i not in sv:
            break

    sv_new = np.array(sv_new)
    return sv_new

def get_w(model):
    sv = model.get_SV() 
    sv_coef = model.get_sv_coef()

    w = np.zeros((len(sv[0]) - 1))
    for i in range(len(sv)):
        w += sv_coef[i] * sv_to_numpy(sv[i])
    
    length = np.sqrt(np.sum(w ** 2))
    return length

def get_w_length_by_kernel(model, kernel):
    sv_dict = model.get_SV()
    sv = [sv_to_numpy(i) for i in sv_dict]
    sv_coef_ = model.get_sv_coef()
    sv_coef = [i[0] for i in sv_coef_]
    sv_num = len(sv)
    w_square = 0
    
    for n in range(sv_num):
        for m in range(sv_num):
            w_square += sv_coef[n] * sv_coef[m] * kernel(sv[n], sv[m])
    
    w_length = np.sqrt(w_square)
    return w_length

def mutiply_w_in_Z(model, kernel, vector):
    sv_dict = model.get_SV()
    sv_array= [sv_to_numpy(sv) for sv in sv_dict]
    sv_coef = model.get_sv_coef()
    sv_num = len(sv_coef)
    value = 0
    for i in range(sv_num):
       value += sv_coef[i][0] * kernel(sv_array[i], vector)

    return value 

def get_first_free_sv(model, c):
    sv_dict = model.get_SV()
    sv_array = [sv_to_numpy(sv) for sv in sv_dict]
    sv_coef = model.get_sv_coef()
    sv_ids = model.get_sv_indices()
    sv_num = len(sv_coef)
    for i in range(sv_num):
        coef = abs(sv_coef[i][0])
        if coef < c:
            return (sv_ids[i], coef, sv_array[i])

def problem13():
    C_log_values = [-5, -3, -1, 1, 3]
    C_values = [(10** c) for c in C_log_values]

    y_train, x_train = read_data(train_data_path)    
    y_train[y_train != 2] = -1
    y_train[y_train == 2] = 1
    
    w_lengths = []
    prob = svm_problem(y_train, x_train)

    for c in C_values:
        print('Experiment with cost value:', c)
        c_param = ' -c ' + str(c)
        param = svm_parameter('-s 0 -t 0' + c_param)
        model = svm_train(prob, param)
        w = get_w(model)
        length = np.sqrt(np.sum(w ** 2))
        w_lengths.append(length)
        print('------------------------------')

    # w_lengths: [1.0336148354311123e-06, 0.0001033614835443807, 0.00033240860949808776, 0.0005929653696404622, 0.018203533288983866]
    x_axis = [str(i) for i in C_log_values]
    plt.title('Problem 13')
    plt.xlabel('C log10 value')
    plt.ylabel('W length')
    plt.plot(x_axis, w_lengths)
    plt.show()

    
def problem14():
    C_log_values = [-5, -3, -1, 1, 3]
    C_values = [(10** c) for c in C_log_values]

    y_train, x_train = read_data(train_data_path)
    y_train[y_train != 4] = -1
    y_train[y_train == 4] = 1

    E_in = []
    prob = svm_problem(y_train, x_train)

    for c in C_values:
        print('Experiment with cost value:', c)
        c_param = ' -c ' + str(c)
        param = svm_parameter('-s 0 -t 1 -g 1 -r 1 -d 2' + c_param)
        model = svm_train(prob, param)
        _, p_acc, _ = svm_predict(y_train, x_train, model)
        
        e = 1 - p_acc[0]/100
        E_in.append(e)
        print('--------------------------')
    
    print(E_in) # [0.08942531888629812, 0.08942531888629812, 0.08942531888629812, 0.08942531888629812, 0.25456041695240694]
    x_axis = [str(i) for i in C_log_values]
    plt.title('Problem 14')
    plt.xlabel('C log10 value')
    plt.ylabel('Ein rate')
    plt.plot(x_axis, E_in)
    plt.show()

def problem15():
    def kernel(x1, x2):
        gamma = 80
        dis_square = np.sum((x1 - x2) ** 2)
        value = np.exp(-gamma * dis_square)
        return value

    C_log_values = [-2, -1, 0, 1, 2]
    C_values = [(10** c) for c in C_log_values]

    y_train, x_train = read_data(train_data_path)
    y_train[y_train != 0] = -1
    y_train[y_train == 0] = 1

    distance = []
    prob = svm_problem(y_train, x_train)

    for c in C_values:
        print('Experiment with cost value:', c)
        c_param = ' -c ' + str(c)
        param = svm_parameter('-s 0 -t 2 -g 80' + c_param)
        model = svm_train(prob, param)
        w_len = get_w_length_by_kernel(model, kernel)
        dis = 1 / w_len
        distance.append(dis)
        print('-----------------------------')

    print(distance) # [0.7686419377676792, 0.1356510814220979, 0.08933322084080739, 0.04385618719847234, 0.014952905940466308]
    x_axis = [str(i) for i in C_log_values]
    plt.title('Problem 15')
    plt.xlabel('log10(C) value')
    plt.ylabel('distance')
    plt.plot(x_axis, distance)
    plt.show()

def problem16():
    gamma_log_values = [-2, -1, 0, 1, 2]
    gamma_values = [(10 ** i) for i in gamma_log_values]

    y_all, x_all = read_data(train_data_path)
    y_all[y_all != 0] = -1
    y_all[y_all == 0] = 1
    data_num = len(y_all)

    best_count = [0, 0, 0, 0, 0]
    repeat = 100
    for _ in range(repeat):
        valid = np.zeros((data_num))
        sample = random.choice(data_num, 1000, replace= False)
        valid[sample] = 1
        y_train, x_train = y_all[valid == 0], x_all[valid == 0]
        y_valid, x_valid = y_all[valid == 1], x_all[valid == 1]
        
        E_vals = []
        prob = svm_problem(y_train, x_train)
        for g in gamma_values:
            g_param = ' -g ' + str(g)
            param = svm_parameter('-s 0 -t 2 -c 0.1' + g_param)
            model = svm_train(prob, param)
            _, p_acc, _ = svm_predict(y_valid, x_valid, model)
            e = 1 - p_acc[0]/100
            E_vals.append(e)

        id_min = E_vals.index(min(E_vals))
        best_count[id_min] += 1
    
    x_axis = [str(i) for i in gamma_log_values]
    print(best_count) # [0, 0, 5, 73, 22]
    plt.title('Problem 16')
    plt.xlabel('log10(Gamma)')
    plt.ylabel('Best count')
    plt.bar(x_axis, best_count)
    plt.show()

def test_libsvm():
    y, x = svm_read_problem('libsvm-3.23/heart_scale')
    model = svm_train(y[:200], x[:200], '-c 5')
    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], model)

def test_qp():
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = np.dot(M.T, M)  # quick way to build a symmetric matrix
    q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))
    sol = solve_qp(P, q, G, h)
    print('QP solution:', sol)

if __name__ == '__main__':
    prob_num = int(sys.argv[1])
    if prob_num == 13:
        problem13()
    elif prob_num == 14:
        problem14()
    elif prob_num == 15:
        problem15()
    elif prob_num == 16:
        problem16()
    else:
        print('Wrong problem number specified ! (13 ~ 16)')