import logistic_regression.logistic_regression as lr
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
    theta = [0., 0., 0.]
    x = numpy.mat(x)  # turn x,y,theta to mat
    y = numpy.mat(y).T
    theta = numpy.mat(theta).T
    theta, theta_list, cost_list = lr.update_GD(theta, x, y, alpha=0.000015,
                                                thre=0.00001)  # update using GD, update about 377000 times
    # theta, min_cost, theta_list, cost_list = lr.update_SGD(theta, x, y, alpha=0.0005, times=100000,
    #                                                        num_r=10)  # update using SGD_OP
    # theta, theta_list, cost_list = lr.update_Newton(theta, x, y, times=5)  # update using Newton method
    print('The time of iteration:', len(cost_list))
    print('The parameters:', theta.T.tolist())
    print('The cost:', cost_list[-1])
    # print('The cost:', min_cost)    # update using SGD_OP
    figure.pic_2(theta_list, cost_list, x, y, interval=5000)  # update using GD
    # figure.pic_3(theta_list, cost_list, x, y, interval=300)  # update using SGD
    # figure.pic_2(theta_list, cost_list, x, y, interval=1)  # update using Newton method
