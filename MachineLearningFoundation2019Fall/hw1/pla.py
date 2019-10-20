import numpy as np

def get_data(file):
    data_raw = np.genfromtxt(file)
    data_x = data_raw[:, :-1]
    data_y = data_raw[:, -1]
    const = np.ones((data_x.shape[0])).reshape(-1, 1)
    data_x = np.hstack((data_x, const))
    return data_x, data_y

def classify(w, x):
    out = np.sign(w @ x)
    if out == 0:
        out = -1
    return out

def pla(data_x, data_y, random= False, eta= 1):
    data_num, dim = data_x.shape
    if random:
        permute = np.random.permutation(data_num)
        data_x, data_y = data_x[permute], data_y[permute]
    w = np.zeros((dim))
    update_num = 0
    last_update = 0
    idx = 0
    while True:
        if (idx - last_update) == data_num + 1:
            break
        x, y = data_x[idx % data_num], data_y[idx % data_num]
        if classify(w, x) != y:
            w += eta * y * x
            update_num += 1
            last_update = idx
        idx += 1

    return update_num, w

def pla_pocket(data_x, data_y, limit, random= False):
    limit_ = limit
    data_num, dim = data_x.shape
    if random:
        permute = np.random.permutation(data_num)
        data_x, data_y = data_x[permute], data_y[permute]
    w_pocket = np.zeros((dim))
    w_t = np.zeros((dim))
    idx = 0
    while True:
        x, y = data_x[idx % data_num], data_y[idx % data_num]
        if classify(w_t, x) != y:
            w_t += y * x
            limit_ -= 1
            if cls_acc(w_t, data_x, data_y) > cls_acc(w_pocket, data_x, data_y):
                w_pocket = w_t.copy()
                
        idx += 1
        if limit_ == 0:
            break

    return w_pocket

def cls_acc(w, data_x, data_y):
    x, y = data_x, data_y
    pred = np.sign(x @ w)
    pred[pred == 0] = -1
    acc = np.sum(pred == y) / len(y)
    return acc

def test():
    train_x, train_y = get_data("coursera/18_train")
    test_x, test_y   = get_data("coursera/18_test")
    data_x, data_y = get_data("coursera/15_train")    
    limit = 20
    # w = pla_pocket(data_x, data_y, limit, random= True)
    update_num, w = pla(train_x, train_y, random= True, eta= 0.5)
    acc = cls_acc(w, data_x, data_y)
    print("Error rate: {}".format(1 - acc))

def coursera_15():
    data_x, data_y = get_data("coursera/15_train")
    updates = []
    for _ in range(2000):
        update_num, w = pla(data_x, data_y, random= True, eta= 0.5)
        updates.append(update_num)
    print("Average updates: {}".format(np.mean(updates)))

def coursera_18():
    train_x, train_y = get_data("coursera/18_train")
    test_x, test_y   = get_data("coursera/18_test")
    limit = 100
    err_rates = []
    for _ in range(2000):
        w = pla_pocket(train_x, train_y, limit, random= True)
        acc = cls_acc(w, test_x, test_y)
        err_rates.append(1 - acc)
    print("Average error rate: {}".format(np.mean(err_rates)))

if __name__ == "__main__":
    coursera_18()