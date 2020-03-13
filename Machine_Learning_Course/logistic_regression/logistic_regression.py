import optimize_algorithm.GD as gd
import optimize_algorithm.NT as nt
import numpy


def sigmoid_function(z):
    '''
    :param z: the input
    :return: the value of sigmoid function
    '''
    return 1.0 / (1.0 + numpy.exp(-z))


def h(theta, x):
    '''
    :param theta: the parameters
    :param x: the training samples
    :return: the value of logistic regression function
    '''
    return sigmoid_function(x * theta)


def cost_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of cost (the log likelihood 's opposite number)
    '''
    return (y.T * numpy.log(h(theta, x)) + (1 - y).T * numpy.log(1 - h(theta, x))) * -1


def gradient_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of gradient (n*1)
    '''
    return (x.T * (y - h(theta, x))) * -1


def hessian_function(theta, x):
    '''
    :param theta: the parameters
    :param x: the training samples
    :return: the hessian mat of the cost function
    '''
    return ((h(theta, x).T * (h(theta, x) - 1))[0, 0] * (x.T * x)) * -1


def update_GD(theta, x, y, alpha, thre):
    '''
    update parameters by GD
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :param alpha: the update rate
    :param thre: the threshold value
    :return: new parameters , parameters list , cost list
    '''
    return gd.gradient_descent(theta, x, y, alpha, thre, cost_func=cost_function, gradient_func=gradient_function)


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


def update_Newton(theta, x, y, times):
    '''
    update parameters by newton's method
    :param theta: the parameters
    :param x: the training samples
    :param y: the ladel of training samples
    :param times: the iteration times
    :return: new parameters , parameters list , cost list
    '''
    return nt.newton_method(theta, x, y, times, cost_func=cost_function, gradient_func=gradient_function,
                            hessian_function=hessian_function)
