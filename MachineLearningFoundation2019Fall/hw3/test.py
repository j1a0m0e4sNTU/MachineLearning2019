import math 
import numpy as np

def grad(pos):
    u, v = pos
    u_grad = math.exp(u) + v*math.exp(u*v) + 2*u - 2*v -3
    v_grad = 2*math.exp(2*v) + u*math.exp(u*v) - 2*u + 4*v - 2
    return np.array([u_grad, v_grad])

def E(pos):
    u, v = pos
    e = math.exp(u) + math.exp(2*v) + math.exp(u*v) + u**2 - 2*u*v + 2*(v**2) - 3*u - 2*v
    return e

def p7():
    pos = np.zeros((2,))
    for _ in range(5):
        pos -= 0.01 * grad(pos)
    print(E(pos))

def newton(pos):
    u, v = pos
    g = grad(pos)
    hess = np.zeros((2, 2))
    hess[0, 0] = math.exp(u) + (v**2)*math.exp(u*v) + 2
    hess[1, 0] = math.exp(u*v) + u*v*math.exp(u*v) - 2
    hess[0, 1] = math.exp(u*v) + u*v*math.exp(u*v) - 2
    hess[1, 1] = 4*math.exp(2*v) + (u**2)*math.exp(u*v) + 4
    inv = np.linalg.inv(hess)
    return (-1) * (inv @ g)

def p10():
    pos = np.zeros((2,))
    for i in range(5):
        pos += newton(pos)
    print(E(pos))


# p12-14
def gen_data(N):
    x = np.ones((N, 3))
    x[:,1:] = np.random.uniform(low= -1, high= 1, size= (N,2))
    y = np.ones(N)
    y[x[:,1]**2 + x[:,2]**2 < 0.6] = -1
    flip_id = np.random.randint(0, N, int(N*0.1))
    y[flip_id] *= (-1)
    return x, y

def transform(x):
    N = x.shape[0]
    x_trans = np.zeros((N, 6))
    x_trans[:, :3] = x
    x_trans[:, 3] = x[:, 1] * x[:, 2]
    x_trans[:, 4] = x[:, 1]**2
    x_trans[:, 5] = x[:, 2]**2
    return x_trans

def get_error(x, y, w):
    pred = np.sign(x @ w)
    error = 1 - np.sum(pred == y) / len(y)
    return error

def p14():
    N = 1000
    train_x, train_y = gen_data(N)
    train_x = transform(train_x)
    w1 = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
    w2 = np.array([-1, -1.5, 0.08, 0.13, 0.05, 1.5])
    w3 = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
    w4 = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
    w5 = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
    print(get_error(train_x, train_y, w1))
    print(get_error(train_x, train_y, w2))
    print(get_error(train_x, train_y, w3))
    print(get_error(train_x, train_y, w4))
    print(get_error(train_x, train_y, w5))

def p15():
    N = 1000
    train_x, train_y = gen_data(N)
    train_x = transform(train_x)
    w =  np.linalg.inv(train_x.T @ train_x) @ (train_x.T @ train_y)
    test_x, test_y = gen_data(N)
    test_x = transform(test_x)
    print("E in:  {}".format(get_error(train_x, train_y, w)))
    print("E out: {}".format(get_error(test_x, test_y, w)))

if __name__ == "__main__":
    p15()