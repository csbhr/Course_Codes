import perceptron.perceptron as pp
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
    theta = [1., 1., 1.]
    x = numpy.mat(x)  # turn x,y,theta to mat
    y = numpy.mat(y).T
    theta = numpy.mat(theta).T
    theta, min_cost, theta_list, cost_list = pp.update_SGD(theta, x, y, alpha=0.0005, times=100000,
                                                           num_r=10)  # update using SGD_OP
    print('The time of iteration:', len(cost_list))
    print('The parameters:', theta.T.tolist())
    print('The cost:', min_cost)
    figure.pic_3(theta_list, cost_list, x, y, interval=300)  # update using SGD
