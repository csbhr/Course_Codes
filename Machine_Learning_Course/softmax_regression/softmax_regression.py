import optimize_algorithm.GD as gd
import numpy


def Indic(theta, x, y):
    m = x.shape[0]
    c = theta.shape[1]
    indic = numpy.mat(numpy.zeros((m, c)))
    for i in range(m):
        indic[i, int(y[i])] = 1
    return indic


def h(theta, x):
    '''
    :param theta: the parameters
    :param x: the training samples
    :return: the value of softmax regression function, it is a mat of m*c
    '''
    exp_mat = numpy.exp(x * theta)
    sum_exp_vector = numpy.sum(exp_mat, axis=1)
    return exp_mat / sum_exp_vector


def cost_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of cost (the log likelihood 's opposite number)
    '''
    indic = Indic(theta, x, y)
    temp = numpy.multiply(indic, numpy.log(h(theta, x)))
    return numpy.sum(temp) * -1


def gradient_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of gradient (n*c)
    '''
    indic = Indic(theta, x, y)
    return (x.T * (indic - h(theta, x))) * -1


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
