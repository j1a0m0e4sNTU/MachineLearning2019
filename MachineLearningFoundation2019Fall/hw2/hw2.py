import math
import numpy as np
from matplotlib import pyplot as plt

def online_p4():
    d_vc = 50
    delta = 0.05
    N = 5
    def m_H(n):
        return n ** d_vc

    a_bound = math.sqrt((8/N) * math.log((4*m_H(2*N))/delta))
    b_bound = math.sqrt(2 * math.log(2*N*m_H(N))/N) + math.sqrt((2/N) * math.log(1/delta)) + 1/N
    c_bound = math.sqrt((1/N) * math.log(6 * m_H(2*N) / delta) + 1/(N**2)) + 1/N
    d_bound = math.sqrt((1/(2*N - 4)) * (math.log(4/delta) + 2*d_vc*math.log(N)) + (1/(N-2))**2) + 1/(N-2)
    e_bound = math.sqrt((16/N) * math.log(2 * m_H(N) / math.sqrt(delta)))

    print("a: ", a_bound)
    print("b: ", b_bound)
    print("c: ", c_bound)
    print("d: ", d_bound)
    print("e: ", e_bound)

def get_data(path):
    data_all = np.genfromtxt(path)
    data_x = data_all[:, :-1]
    data_y = data_all[:, -1]
    return data_x, data_y

def error(pred, gt):
    return np.sum(pred != gt) / len(gt)

def decision_stump(data_x, data_y):
    """
    x, y are both in shape (data_num, )
    Return (theta, s= +/-1)
    """
    data_len = len(data_x)
    sort_arg = np.argsort(data_x)
    data_x = data_x[sort_arg]
    data_y = data_y[sort_arg]
    thetas = [(data_x[i] + data_x[i+1]) / 2 for i in range(data_len-1)]
    thetas.append(data_x[0] - 1)
    stump = (None, None)
    error_min = 1
    for theta in thetas:
        pred = np.ones(data_len) * (-1)
        pred[data_x > theta] = 1
        err = error(pred, data_y)
        if  err < error_min:
            error_min = err
            stump = (theta, 1)
        pred = np.ones(data_len) * (-1)
        pred[data_x < theta] = 1
        err = error(pred, data_y)
        if  err < error_min:
            error_min = err
            stump = (theta, -1)
    return stump

def online_p19_p20():
    train_x, train_y = get_data("train.dat")
    test_x , test_y  = get_data("test.dat")
    train_len, test_len = len(train_y), len(test_y)

    dim = None 
    stump = None
    E_in = 1
    for d in range(train_x.shape[1]):
        ds = decision_stump(train_x[:, d], train_y)
        pred = np.ones(train_len) * (ds[1])
        pred[train_x[:, d] < ds[0]] *= (-1)
        err = error(pred, train_y)
        if err < E_in:
            E_in = err
            dim = d
            stump = ds
    
    test_pred = np.ones(test_len) * stump[1]
    test_pred[test_x[:, dim] < stump[0]] *= (-1)
    E_out = error(test_pred, test_y)

    print("E in:  {}".format(E_in))
    print("E_out: {}".format(E_out))

def E_out(s, theta):
    return 0.5 + 0.3 * s * (min(abs(theta), 1) - 1)

def plot_hist(data_size, fig_name):
    error_distance = []
    error_in  = []
    error_out = []
    for _ in range(1000):
        x = np.random.rand(data_size) * 2 - 1
        y = np.ones(data_size)
        y[x < 0] = -1
        noise_idx = np.random.choice(data_size, int(0.2 * data_size), replace= False)
        y[noise_idx] *= -1

        theta, s = decision_stump(x, y)
        pred = np.ones(data_size) * s
        pred[x < theta] *= (-1)
        e_in  = error(pred, y)
        e_out = E_out(s, theta)
        e_diff = e_in - e_out
        error_distance.append(e_diff)
        error_in.append(e_in)
        error_out.append(e_out)
    
    print("E_in: {}".format(np.mean(error_in)))
    print("E_out: {}".format(np.mean(error_out)))
    plt.title("E_in: {:.3f} E_out: {:.3f}".format(np.mean(error_in), np.mean(error_out)))
    plt.hist(error_distance)
    plt.savefig(fig_name)
    plt.close()

def hw_p7_p8():
    plot_hist(20, "imgs/p7.png")
    plot_hist(2000, "imgs/p8.png")

def test():
    x, y = get_data("test.dat")
    print(x.shape)
    print(y.shape)

if __name__ == "__main__":
    hw_p7_p8()