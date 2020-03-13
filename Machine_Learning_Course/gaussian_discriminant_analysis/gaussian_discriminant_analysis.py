import numpy


def bernoulli_distribution(phi, y):
    '''
    :param phi: the parameter of bernoulli distribution (R)
    :param y: the input value (m*1)
    :return: the probability of y in bernoulli distribution (m*1)
    '''
    return numpy.multiply(numpy.power(phi, y), numpy.power(1 - phi, 1 - y))


def gaussian_distribution(mu, sigma, x):
    '''
    :param mu: the mean of gaussian distribution (1*n)
    :param sigma: the variance of gaussian distribution (n*n)
    :param x: the input value (m*n)
    :return: the probability of x in gaussian distribution (m*1)
    '''
    n = x.shape[1]
    ele = numpy.exp((x - mu) * sigma.I * (x - mu).T * (-1 / 2))
    ele = ele.diagonal().T  # turn m*m to m*1 by getting diagonal
    mole = numpy.power(2 * numpy.pi, n / 2) * numpy.power(numpy.linalg.det(sigma), 1 / 2)
    return ele / mole


def probability_y(phi, y):
    '''
    :param phi: the parameter of bernoulli distribution (R)
    :param y: the input value (m*1)
    :return: the probability of y in bernoulli distribution (m*1)
    '''
    return bernoulli_distribution(phi, y)


def probability_x_y0(mu0, sigma, x):
    '''
    :param mu0: the mean of gaussian distribution given y=0 (1*n)
    :param sigma: the variance of gaussian distribution (n*n)
    :param x: the input value (m*n)
    :return: the probability of x given y=0 in gaussian distribution (m*1)
    '''
    return gaussian_distribution(mu0, sigma, x)


def probability_x_y1(mu1, sigma, x):
    '''
    :param mu1: the mean of gaussian distribution given y=1 (1*n)
    :param sigma: the variance of gaussian distribution (n*n)
    :param x: the input value (m*n)
    :return: the probability of x given y=1 in gaussian distribution (m*1)
    '''
    return gaussian_distribution(mu1, sigma, x)


def h(phi, mu0, mu1, sigma, x):
    '''
    :param phi: the parameter of bernoulli distribution (R)
    :param mu0: the mean of gaussian distribution given y=0 (1*n)
    :param mu1: the mean of gaussian distribution given y=1 (1*n)
    :param sigma: the variance of gaussian distribution (n*n)
    :param x: the input value (m*n)
    :return: the prediction of p(y=0|x) and p(y=1|x) (m*2)
    '''
    y0_mat = numpy.zeros([x.shape[0], 1])
    y1_mat = numpy.ones([x.shape[0], 1])
    pro_y0 = probability_y(phi, y0_mat)  # m*1
    pro_y1 = probability_y(phi, y1_mat)  # m*1
    pro_x_y0 = probability_x_y0(mu0, sigma, x)  # m*1
    pro_x_y1 = probability_x_y1(mu1, sigma, x)  # m*1
    pro_x = numpy.multiply(pro_x_y0, pro_y0) + numpy.multiply(pro_x_y1, pro_y1)
    pro_y0_x = numpy.multiply(pro_x_y0, pro_y0) / pro_x
    pro_y1_x = numpy.multiply(pro_x_y1, pro_y1) / pro_x
    return numpy.vstack((pro_y0_x.T, pro_y1_x.T)).T


def indicator_function(y, value):
    '''
    :param y: the label of training samples (m*1)
    :param value: the indicator value
    :return: the indicator mat (m*1)
    '''
    m = y.shape[0]
    indicator = numpy.mat(numpy.zeros((m, 1)))
    for i in range(m):
        if y[i] == value:
            indicator[i, 0] = 1
    return indicator


def compute_parameters(x, y):
    '''
    :param x: the training samples (m*n)
    :param y: the label of training samples (m*1)
    :return: the parameters( phi (R), mu0 (1*n), mu1 (1*n), sigma(n*n) ) of the model
    '''
    m = x.shape[0]
    indic_y0 = indicator_function(y, 0)
    indic_y1 = indicator_function(y, 1)
    num_y0, num_y1 = numpy.sum(indic_y0), numpy.sum(indic_y1)
    phi = num_y1 / m  # compute phi
    mu0 = numpy.sum(numpy.multiply(indic_y0, x), axis=0) / num_y0  # compute mu0
    mu1 = numpy.sum(numpy.multiply(indic_y1, x), axis=0) / num_y1  # compute mu1
    mu0_mat = numpy.multiply(indic_y0, mu0.repeat(m, axis=0))
    mu1_mat = numpy.multiply(indic_y1, mu1.repeat(m, axis=0))
    mu_mat = mu0_mat + mu1_mat
    sigma = (x - mu_mat).T * (x - mu_mat) / m  # compute sigma
    return phi, mu0, mu1, sigma


def verify_function(phi, mu0, mu1, sigma, x, y):
    '''
    :param phi: the parameter of bernoulli distribution (R)
    :param mu0: the mean of gaussian distribution given y=0 (1*n)
    :param mu1: the mean of gaussian distribution given y=1 (1*n)
    :param sigma: the variance of gaussian distribution (n*n)
    :param x: the verifying samples (m*n)
    :param y: the label of verifying samples (m*1)
    :return: verify the accuracy with the verifying samples set
    '''
    m = x.shape[0]  # the number of verifying samples
    y_ = h(phi, mu0, mu1, sigma, x)
    y_ = numpy.argmax(y_, axis=1)
    diff = numpy.abs(y - y_)
    correct_num = 0
    for i in range(m):
        if diff[i] < 0.5:
            correct_num = correct_num + 1
    return float(correct_num) / float(m)
