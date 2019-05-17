import numpy as np

class Node():
    def __init__(self):
        self.feature_id = None
        self.threshold = None
        self.left_node = None
        self.right_node = None
        self.hypothesis = None

    def get_gini_index(self, xy_data):
        y_data = xy_data[:, -1]
        total_num = xy_data.shape[0]
        labels = np.unique(y_data)
        gini_index = 1
        for label in labels:
            gini_index -= (np.sum(y_data == label) / total_num) ** 2
        return gini_index

    def get_sorted_data(self, xy_data, index):
        data = xy_data.copy()
        data_num = data.shape[0]
        for i in range(1, data_num):
            data_hold = data[i].copy()
            j = i - 1 
            while True:
                if data[j, index] > data_hold[index]:
                    data[j + 1] = data[j]
                    if j == 0:
                        data[0] = data_hold
                        break
                    j -= 1
                else:
                    data[j + 1] = data_hold
                    break
        return data

    def train(self, xy_data):
        hypothesis = np.sign(np.sum(xy_data[:, -1]))
        self.hypothesis = hypothesis if hypothesis != 0 else 1
        if (len(np.unique(xy_data[:, -1])) == 1):
            return None

        Gini_index = 1
        data_optim = (None, None)
        feature_num = xy_data.shape[1] - 1
        data_num = xy_data.shape[0]
        for f_id in range(feature_num):
            data = self.get_sorted_data(xy_data, f_id)
            for d_id in range(1, data_num):
                data_part1 = data[:d_id]
                data_part2 = data[d_id:]
                gini = self.get_gini_index(data_part1) + self.get_gini_index(data_part2)
                if gini < Gini_index:
                    Gini_index = gini
                    data_optim = (data_part1, data_part2)
                    self.feature_id = f_id
                    self.threshold = (data[d_id, f_id] + data[d_id -1, f_id])/2
        
        return data_optim
        ##
        if self.get_gini_index(xy_data) < Gini_index:
            return None
        else:
            return data_optim

    def predict(self, x_data):
        if (self.left_node == None and self.right_node == None):
            return self.hypothesis
        elif x_data[self.feature_id] < self.threshold:
            return self.left_node.predict(x_data)
        else:
            return self.right_node.predict(x_data)

####################
class DecisionTree():
    def __init__(self, data, height= None):
        self.root = Node()
        data_split = self.root.train(data)
        self.train(self.root, data_split, height)

    def train(self, node, data_split, height_remain):
        if data_split == None:
            return 
        data_left, data_right = data_split
        node_left, node_right = Node(), Node()
        data_split_left = node_left.train(data_left)
        data_split_right = node_right.train(data_right)
        node.left_node, node.right_node = node_left, node_right
        if height_remain:
            if height_remain > 1:
                self.train(node_left, data_split_left, height_remain - 1)
                self.train(node_right, data_split_right, height_remain - 1)
        else:
            self.train(node_left, data_split_left, None)
            self.train(node_right, data_split_right, None)

    def predict(self, x_data):
        return self.root.predict(x_data)

    def predict_all(self, x_data_all):
        prediction = []
        size = x_data_all.shape[0]
        for i in range(size):
            prediction.append(self.predict(x_data_all[i]))
        return np.array(prediction)

    def show(self):
        pass

def test_node():
    node = Node()
    data = np.random.rand(5, 5)
    data[:, -1] = 1
    print(data)
    result = node.train(data)
    print(result)
    print(node.feature_id)
    print(node.threshold)
    print(node.hypothesis)

def test_decision_tree():
    data = np.random.rand(100, 5)
    data -= 0.5
    data[:, -1] = np.sign(data[:, -1])
    dt = DecisionTree(data, 80)
    
    print(data)
    error_count = np.sum(np.abs(np.sign(data[:, -1] - dt.predict_all(data[:, :-1]))))
    print("Error count:", error_count)

if __name__ == '__main__':
    test_decision_tree()