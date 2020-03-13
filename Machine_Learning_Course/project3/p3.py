import softmax_regression.softmax_regression as sr
import matplotlib_helper.figure as figure
import numpy


def get_data(path):
    data = numpy.loadtxt(path)
    return data


def action():
    x_ = get_data("data_files/ex4x.dat")
    y = get_data("data_files/ex4y.dat")
    x = []
    for i in range(len(x_)):
        x.append([1, x_[i][0], x_[i][1]])
    theta = [[0., 0.],
             [0., 0.],
             [0., 0.]]
    x = numpy.mat(x)  # turn x,y,theta to mat
    y = numpy.mat(y).T
    theta = numpy.mat(theta)
    theta, theta_list, cost_list = sr.update_GD(theta, x, y, alpha=0.000008,
                                                thre=0.00001)  # update using GD, update about 365000 times
    print('The time of iteration:', len(cost_list))
    print('The parameters:', theta.T.tolist())
    print('The cost:', cost_list[-1])
    figure.pic_2(theta_list, cost_list, x, y, interval=5000)  # update using GD
