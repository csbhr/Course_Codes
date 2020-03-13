import optimize_algorithm.GD as gd


def h(theta, x):
    '''
    :param theta: the parameters
    :param x: the training samples
    :return: the values linear regression function
    '''
    return x * theta


def cost_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of cost
    '''
    return ((h(theta, x) - y).T * (h(theta, x) - y)) / 2


def gradient_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :return: the value of gradient (n*1)
    '''
    return x.T * (h(theta, x) - y)


def update_GD(theta, x, y, alpha, thre):
    '''
    update parameters
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :param alpha: the update rate
    :param thre: the threshold value
    :return: new parameters , parameters list , cost list
    '''
    return gd.gradient_descent(theta, x, y, alpha, thre, cost_func=cost_function, gradient_func=gradient_function)
