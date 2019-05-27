import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossManager():
    def __init__(self):
        self.interval_list  = []
        self.train_mse_list = []
        self.train_wmae_list= []
        self.train_nae_list = []
        self.valid_mse_list = []
        self.valid_wmae_list= []
        self.valid_nae_list = []

    def record(self, interval, results):
        # results: (train_mse, train_wmae, train_nae, valid_mse, valid_wmae, valid_nae)
        self.interval_list.append(interval)
        train_mse, train_wmae, train_nae, valid_mse, valid_wmae, valid_nae = results
        self.train_mse_list.append(train_mse)
        self.train_wmae_list.append(train_wmae)
        self.train_nae_list.append(train_nae)
        self.valid_mse_list.append(valid_mse)
        self.valid_wmae_list.append(valid_wmae)
        self.valid_nae_list.append(valid_nae)

    def get_train_mse(self):
        return self.train_mse_list
    
    def get_train_wmae(self):
        return self.train_wmae_list
    
    def get_train_nae(self):
        return self.train_nae_list

    def get_valid_mse(self):
        return self.valid_mse_list

    def get_valid_wmae(self):
        return self.valid_wmae_list 

    def get_valid_nae(self):
        return self.valid_nae_list   
    
    def plot_mse(self, title, x_label, fig_name):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('MSE')
        plt.plot(self.interval_list, self.train_mse_list, c= 'b')
        plt.plot(self.interval_list, self.valid_mse_list, c= 'r')
        plt.savefig(fig_name)
        plt.close()

    def plot_wmae(self, title, x_label, fig_name):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('WMAE')
        plt.plot(self.interval_list, self.train_wmae_list, c= 'b')
        plt.plot(self.interval_list, self.valid_wmae_list, c= 'r')
        plt.savefig(fig_name)
        plt.close()

    def plot_nae(self, title, x_label, fig_name):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('NAE')
        plt.plot(self.interval_list, self.train_nae_list, c= 'b')
        plt.plot(self.interval_list, self.valid_nae_list, c= 'r')
        plt.savefig(fig_name)
        plt.close()

    def plot_all(self, title, x_label, fig_name):
        plt.subplot(3, 1, 1)
        plt.title(title)
        plt.ylabel('MSE')
        plt.plot(self.interval_list, self.train_mse_list, c= 'b')
        plt.plot(self.interval_list, self.valid_mse_list, c= 'r')

        plt.subplot(3, 1, 2)
        plt.ylabel('WMAE')
        plt.plot(self.interval_list, self.train_wmae_list, c= 'b')
        plt.plot(self.interval_list, self.valid_wmae_list, c= 'r')

        plt.subplot(3, 1, 3)
        plt.ylabel('NAE')
        plt.xlabel(x_label)
        plt.plot(self.interval_list, self.train_nae_list, c= 'b')
        plt.plot(self.interval_list, self.valid_nae_list, c= 'r')

        plt.savefig(fig_name)

def test():
    import random
    loss_manager = LossManager()
    
    for i in range(20):
        results = [random.random() for _ in range(6)]
        loss_manager.record(i, results)
    
    loss_manager.plot_all('test', 'x', 'test.png')

if __name__ == '__main__':
    test()