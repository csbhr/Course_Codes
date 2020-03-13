import numpy


def bernoulli_distribution(phis, x):
    '''
    :param phis: the parameter of bernoulli distribution (1*n)
    :param x: the input value (m*n)
    :return: the probability of y in bernoulli distribution (m*n)
    '''
    return numpy.multiply(numpy.power(phis, x), numpy.power(1 - phis, 1 - x))


def probability_y(phi, y):
    '''
    :param phi: P(y=1) (1*1)
    :param y: the input value (m*1)
    :return: P(y) (m*1)
    '''
    return bernoulli_distribution(phi, y)


def probability_x_y0(phis0, x):
    '''
    :param phis0: phis[i] is p(xi=1|y=0) (1*n)
    :param x: the input value (m*n)
    :return: p(x|y=0) (m*1)
    '''
    pro_res = bernoulli_distribution(phis0, x)  # (m*n)
    return numpy.prod(pro_res, axis=1)  # (m*1)


def probability_x_y1(phis1, x):
    '''
    :param phis1: phis[i] is p(xi=1|y=1) (1*n)
    :param x: the input value (m*n)
    :return: p(x|y=1) (m*1)
    '''
    pro_res = bernoulli_distribution(phis1, x)  # (m*n)
    return numpy.prod(pro_res, axis=1)  # (m*1)


def h(phi, phis0, phis1, x):
    '''
    :param phi: P(y=1) (1*1)
    :param phis0: P(x|y=0) (1*n)
    :param phis1: P(x|y=1) (1*n)
    :param x: the input value (m*n)
    :return: the prediction of p(y=0|x) and P(y=1|x) (m*2)
    '''
    y0_mat = numpy.zeros([x.shape[0], 1])
    y1_mat = numpy.ones([x.shape[0], 1])
    pro_y0 = probability_y(phi, y0_mat)  # m*1
    pro_y1 = probability_y(phi, y1_mat)  # m*1
    pro_x_y0 = probability_x_y0(phis0, x)  # m*1
    pro_x_y1 = probability_x_y1(phis1, x)  # m*1
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
    :return: the parameters( phi (1*1), phis0 (1*n), phis0 (1*n) ) of the model
    '''
    m, k = x.shape[0], x.shape[1]  # the number of samples , the dimension of sample
    indic_y0 = indicator_function(y, 0)
    indic_y1 = indicator_function(y, 1)
    num_y0, num_y1 = numpy.sum(indic_y0), numpy.sum(indic_y1)
    phi = numpy.mat([num_y1 / m])  # compute phi (1*1)
    phis0 = (numpy.sum(numpy.multiply(indic_y0, x), axis=0) + 1) / (
            num_y0 + k)  # compute mu0 (1*n) with laplace smoothing
    phis1 = (numpy.sum(numpy.multiply(indic_y1, x), axis=0) + 1) / (
            num_y0 + k)  # compute mu1 (1*n) with laplace smoothing
    return phi, phis0, phis1


def verify_function(phi, phis0, phis1, x, y):
    '''
    :param phi: P(y=1) (1*1)
    :param phis0: P(x|y=0) (1*n)
    :param phis1: P(x|y=1) (1*n)
    :param x: the verifying samples (m*n)
    :param y: the label of verifying samples (m*1)
    :return: verify the accuracy with the verifying samples set
    '''
    m = x.shape[0]  # the number of verifying samples
    y_ = h(phi, phis0, phis1, x)
    y_ = numpy.argmax(y_, axis=1)
    diff = numpy.abs(y - y_)
    correct_num = 0
    for i in range(m):
        if diff[i] < 0.5:
            correct_num = correct_num + 1
    return float(correct_num) / float(m)
