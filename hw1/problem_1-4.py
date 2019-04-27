from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np

x_raw = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]) 
y_label = np.array([-1, -1, -1, 1, 1, 1, 1]) 
colors = []
for i in y_label:
    c = 'blue' if y_label[i] == 1 else 'red'
    colors.append(c)

def p1():
    def transform(x):
        x1, x2 = x
        e1 = 2 * (x2 ** 2) -  4 * x1 + 2
        e2 = (x1 ** 2) - 2 * x2 - 3
        return [e1, e2]
        
    x_transform = np.array([transform(x) for x in x_raw])
    
    # plt.scatter(x_raw[:, 0], x_raw[:, 1], c=colors, s = 10)

    svm = SVC(C=100, kernel= 'linear')
    svm.fit(x_transform, y_label)
    support_id = svm.support_
    support_vectors = svm.support_vectors_
    coef = svm.dual_coef_
    w = 0; b = 0
    support_num = len(support_id)
    print('support vectors:\n',support_vectors)
    print('coefficients:',coef[0])
    
    for i in range(support_num):
        w += coef[0][i] * support_vectors[i]
    for i in range(support_num):
        b += (1 / y_label[support_id[i]]) - (w @ support_vectors[i])
    b /= support_num
    
    print('w:',w)
    print('b:',b)

def p2():
    def kernel(x1, x2):
        return (1 + x1 @ x2) ** 2
    
    svm = SVC(C=100, kernel= 'poly', degree= 2, gamma= 1, coef0= 1)
    svm.fit(x_raw, y_label)

    support_id = svm.support_
    support_vecotors = svm.support_vectors_
    coef = svm.dual_coef_[0]

    s_num = len(support_id)
    s_id = 4
    xs = support_vecotors[s_id]
    b = y_label[support_id[s_id]]
    for i in range(s_num):
        b -= coef[i] *  kernel(support_vecotors[i], xs)
    
    c_1 = np.array([1., 1., 0., 4., 4.]) @ coef
    c_2 = coef[2]
    c_3 = -b + coef[2]
    return c_1, c_2, c_3


def plot_p1_curve():
    curve_x2_p1 = np.arange(-2, 2, 0.001)
    curve_x1_p1 = (1/4) * (2 * (curve_x2_p1 ** 2) - 3)
    plt.scatter(x_raw[:, 0], x_raw[:, 1], c=colors, s = 100)
    plt.scatter(curve_x1_p1, curve_x2_p1, c='green', s = 0.1)
    # plt.show()
    plt.title('P1 nonlinear curve')
    plt.savefig('img/p1_curve.png')
    plt.close()

def plot_p3_curve():
    c_1, c_2, c_3 = p2()
    curve_x2_p3 = np.arange(-np.sqrt(c_3/c_1), np.sqrt(c_3/c_1), 0.001)
    curve_x1_up_p3 = np.sqrt(c_3 - c_1 * (curve_x2_p3 ** 2))/c_2 + 1
    curve_x1_down_p3 = -np.sqrt(c_3 - c_1 * (curve_x2_p3 ** 2))/c_2 + 1
    plt.scatter(x_raw[:, 0], x_raw[:, 1], c=colors, s = 100)
    plt.scatter(curve_x1_up_p3 , curve_x2_p3, c='green', s = 0.1)
    plt.scatter(curve_x1_down_p3 , curve_x2_p3, c='green', s = 0.1)
    plt.title('P3 nonlinear curve')
    plt.savefig('img/p3_curve.png')
    plt.close()
    
if __name__ == '__main__':
    plot_p1_curve()
    plot_p3_curve()