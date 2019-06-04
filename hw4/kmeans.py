import sys
import numpy as np
from matplotlib import pyplot as plt
from knn import get_distance_matrix

class KMeans():
    def __init__(self, k):
        self.k = k
        self.centers = []

    def fit(self, data):
        sample_ids = np.random.choice(data.shape[0], self.k, replace= False)
        centers = data[sample_ids]
        cluster_ids = np.zeros(data.shape[0])
        while True:
            distance_matrix = get_distance_matrix(centers, data)
            new_cluster_ids = np.argmin(distance_matrix, 1)
            if np.any(new_cluster_ids != cluster_ids):
                cluster_ids = new_cluster_ids
                for i in range(self.k):
                    centers[i] = np.mean(data[cluster_ids == i], 0)
            else:
                break

        self.centers = centers

    def evaluate(self, data):
        distance_matrix = get_distance_matrix(self.centers, data)
        cluster_ids = np.argmin(distance_matrix, 1)
        error = 0
        for i in range(self.k):
            cluster_diff = data[cluster_ids == i] - self.centers[i]
            error += np.sum(cluster_diff ** 2)
        return error

def get_data():
    data = np.genfromtxt('nolabel.dat')
    return data

def problem_15():
    data = get_data()
    k_list = [2, 4, 6, 8, 10]
    error_mean_list = []
    repeat = 500
    for k in k_list:
        error = 0
        kmeans = KMeans(k)
        for _ in range(repeat):
            kmeans.fit(data)
            error += kmeans.evaluate(data)
        error_mean_list.append(error / repeat)

    plt.title('Problem 15')
    plt.xlabel('K value')
    plt.ylabel('Mean of Ein')
    plt.scatter(k_list, error_mean_list)
    plt.savefig('img/problem_15.png')

def problem_16():
    data = get_data()
    k_list = [2, 4, 6, 8, 10]
    error_var_list = []
    repeat = 500
    for k in k_list:
        error = []
        kmeans = KMeans(k)
        for _ in range(repeat):
            kmeans.fit(data)
            error.append(kmeans.evaluate(data)) 
        error_var_list.append(np.var(error))

    plt.title('Problem 16')
    plt.xlabel('K value')
    plt.ylabel('Variance of Ein')
    plt.scatter(k_list, error_var_list)
    plt.savefig('img/problem_16.png')

def test():
    data = get_data()
    kmeans = KMeans(10)
    kmeans.fit(data)
    print('Done!')

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == '15':
        problem_15()
    elif mode == '16':
        problem_16()
    elif mode == 'test':
        test()
    else:
        print('Error: Wrong argument !')