import numpy as np

# == Y data ==
# data 1 : Penetration rate
# data 2 : Mesh size
# data 3 : Alpha value

x_train_path = '../../data_CSIE_ML/X_train/arr_0.npy'
y_train_path = '../../data_CSIE_ML/Y_train/arr_0.npy'
x_test_path  = '../../data_CSIE_ML/X_test/arr_0.npy'
x_mean_std_path = 'x_mean_std.npy'
y_mean_std_path = 'y_mean_std.npy'
x_sample_path = '../../data_CSIE_ML/X_train/x_sample.npy'
y_sample_path = '../../data_CSIE_ML/Y_train/y_sample.npy'
FEATURE_NUM = 3

def normalize(array, mean, std):
    return (array - mean) / std

def denormalize(array, mean, std):
    return (array * std) + mean

def get_train_data(validation= 0.2, 
                    x_train_path= x_train_path,
                    y_train_path= y_train_path,
                    x_mean_std_path= x_mean_std_path,
                    y_mean_std_path= y_mean_std_path):

    x_train_all = np.load(x_train_path)
    x_mean_std = np.load(x_mean_std_path)
    y_train_all = np.load(y_train_path)
    y_mean_std = np.load(y_mean_std_path)

    x_train_all = normalize(x_train_all, x_mean_std[0], x_mean_std[1])
    y_train_all = normalize(y_train_all, y_mean_std[0], y_mean_std[1])
    
    if 0 < validation < 1:
        cut_size = int((1 - validation) * x_train_all.shape[0])
        x_train, x_valid = x_train_all[:cut_size], x_train_all[cut_size:]
        y_train, y_valid = y_train_all[:cut_size], y_train_all[cut_size:]
        return x_train, y_train, x_valid, y_valid
    else:
        return x_train_all, y_train_all, None, None

def get_sample_data(validation= 0.2, 
                    x_sample_path= x_sample_path,
                    y_sample_path= y_sample_path,
                    x_mean_std_path= x_mean_std_path,
                    y_mean_std_path= y_mean_std_path):

    x_train_all = np.load(x_sample_path)
    x_mean_std = np.load(x_mean_std_path)
    x_mean, x_std = x_mean_std[0, :200], x_mean_std[1, :200]
    y_train_all = np.load(y_sample_path)
    y_mean_std = np.load(y_mean_std_path)
    
    x_train_all = normalize(x_train_all, x_mean, x_std)
    y_train_all = normalize(y_train_all, y_mean_std[0], y_mean_std[1])
    
    cut_size = int((1 - validation) * x_train_all.shape[0])
    x_train, x_valid = x_train_all[:cut_size], x_train_all[cut_size:]
    y_train, y_valid = y_train_all[:cut_size], y_train_all[cut_size:]
    return x_train, y_train, x_valid, y_valid 

def get_test_data():
    x_test_all = np.load(x_test_path)
    x_mean_std = np.load(x_mean_std_path)
    x_test_all = normalize(x_test_all, x_mean_std[0], x_mean_std[1])
    return x_test_all

def write_submission(pred, file_name):
    file = open(file_name, 'w')
    y_mean_std = np.load(y_mean_std_path)
    pred = denormalize(pred, y_mean_std[0], y_mean_std[1])
    data_num = pred.shape[0]
    for i in range(data_num):
        file.write('{},{},{}\n'.format(pred[i, 0], pred[i, 1], pred[i, 2]))
    print('File {} is saved'.format(file_name))

def average_mse(pred, y):
    y_mean_std = np.load(y_mean_std_path)
    pred = denormalize(pred, y_mean_std[0], y_mean_std[1])
    y  = denormalize(y, y_mean_std[0], y_mean_std[1])
    mse = np.sqrt(np.sum((pred - y)**2) / y.shape[0])
    return mse

def wmae_error(pred, y):
    # Track 1 : Weighted Mean Absolute Error
    y_mean_std = np.load(y_mean_std_path)
    y_mean, y_std = y_mean_std[0], y_mean_std[1]
    pred = denormalize(pred, y_mean, y_std)
    y    = denormalize(y, y_mean, y_std)

    weight = np.array([300, 1, 200])
    L1_distence = np.abs(y - pred)
    error = np.sum(L1_distence * weight) / y.shape[0]
    return error

def nae_error(pred, y):
    # Track 2 : Normalized Absolute Error (NAE)
    y_mean_std = np.load(y_mean_std_path)
    y_mean, y_std = y_mean_std[0], y_mean_std[1]
    pred = denormalize(pred, y_mean, y_std)
    y    = denormalize(y, y_mean, y_std)

    L1_distence = np.abs(y - pred)
    error = np.sum(L1_distence / y) / y.shape[0]
    return error

def evaluate(pred, y):
    y_mean_std = np.load(y_mean_std_path)
    y_mean, y_std = y_mean_std[0], y_mean_std[1]
    pred = denormalize(pred, y_mean, y_std)
    y = denormalize(y, y_mean, y_std)
    
    size = pred.shape[0]
    L1 = np.abs(pred - y)
    # WMAE
    weight = np.array([300, 1, 200])
    wmae = np.sum(L1 * weight) / size
    # NAE
    nae = np.sum(L1 / y) / size
    
    return wmae, nae

def test():
    pred = np.random.randn(10, 3)
    y = np.random.rand(10, 3)
    print('MSE: ', average_mse(pred, y))
    print('WMAE:', wmae_error(pred, y))
    print('NAE: ', nae_error(pred, y))

def test2():
    pred = np.random.randn(10, 3)
    y = np.random.randn(10, 3)
    wmae, nae = evaluate(pred, y)
    print('WMAE: {}'.format(wmae))
    print('NAE: {}'.format(nae))

if __name__ == '__main__':
    test2()