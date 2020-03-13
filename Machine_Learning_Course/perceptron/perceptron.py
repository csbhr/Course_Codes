import optimize_algorithm.GD as gd
import numpy


def h(theta, x):
    '''
    :param theta: the parameters
    :param x: the training samples
    :return: the value of perceptron function
    '''
    m = x.shape[0]
    wtx = x * theta
    for i in range(m):
        if (wtx[i] > 0):
            wtx[i] = 1
        else:
            wtx[i] = 0
    return wtx


def cost_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of cost
    '''
    temp = numpy.multiply((h(theta, x) - y), x * theta)
    return numpy.sum(temp)


def gradient_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of gradient (n*1)
    '''
    return x.T * (h(theta, x) - y)


def update_SGD(theta, x, y, alpha, times, num_r):
    '''
    update parameters by SGD
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :param alpha: the update rate
    :param times: the iteration times
    :param num_r: the number of samples we choose randomly each time
    :return: new parameters , parameters list , cost list
    '''
    return gd.stochastic_gradient_descent(theta, x, y, alpha, times, num_r, cost_func=cost_function,
                                          gradient_func=gradient_function)
