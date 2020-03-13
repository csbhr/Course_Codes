import linear_regression.linear_regression as lr
import matplotlib_helper.figure as figure
import numpy


def action():
    x = [[1., 0.], [1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.], [1., 6.], [1., 7.], [1., 8.], [1., 9.], [1., 10.],
         [1., 11.], [1., 12.], [1., 13.]]
    y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
    theta = [0., 0.]
    x = numpy.mat(x)
    y = numpy.mat(y).T
    theta = numpy.mat(theta).T
    theta, theta_list, cost_list = lr.update_GD(theta, x, y, alpha=0.001, thre=0.00001)  # update
    print('The time of iteration:', len(cost_list))
    print('The now cost:', cost_list[-1])
    print('The parameters:', theta.T.tolist())
    figure.pic_1(theta_list, cost_list, x, y, interval=10)
