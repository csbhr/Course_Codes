import numpy


def probability_y(phi, y):
    '''
    :param phi: P(y=1)=phi (R)
    :param y: the input value (list m*1)
    :return: P(y) (list m*1)
    '''
    y = numpy.mat(y)  # turn list to mat
    pro_mat = numpy.multiply(numpy.power(phi, y), numpy.power(1 - phi, 1 - y))  # compute P(y), result is a mat
    return pro_mat.tolist()  # turn mat to list


def dict_produce(dictionary, x):
    '''
    :param dictionary: dict probability (list 1*n)
    :param x: sample (list m*?)
    :return: (list m*1)
    '''
    result = []
    for i in x:
        prod = 1.0
        for j in i:
            prod = prod * dictionary[j]
        result.append([prod])
    return result


def probability_x_y0(phis0, x):
    '''
    :param phis0: phis[i] is p(xi=1|y=0) (list 1*n)
    :param x: the input value (list m*?)
    :return: p(x|y=0) (list m*1)
    '''
    return dict_produce(phis0, x)


def probability_x_y1(phis1, x):
    '''
    :param phis1: phis[i] is p(xi=1|y=0) (list 1*n)
    :param x: the input value (list m*?)
    :return: p(x|y=0) (list m*1)
    '''
    return dict_produce(phis1, x)


def h(phi, phis0, phis1, x):
    '''
    :param phi: P(y=1) (R)
    :param phis0: P(x|y=0) (list 1*n)
    :param phis1: P(x|y=1) (list 1*n)
    :param x: the input value (list m*?)
    :return: the prediction of p(y=0|x) and P(y=1|x) (list m*2)
    '''
    y0_list = numpy.zeros([len(x), 1]).tolist()
    y1_list = numpy.ones([len(x), 1]).tolist()
    pro_y0 = probability_y(phi, y0_list)  # list m*1
    pro_y1 = probability_y(phi, y1_list)  # list m*1
    pro_x_y0 = probability_x_y0(phis0, x)  # list m*1
    pro_x_y1 = probability_x_y1(phis1, x)  # list m*1
    pro_y0_mat = numpy.mat(pro_y0)
    pro_y1_mat = numpy.mat(pro_y1)
    pro_x_y0_mat = numpy.mat(pro_x_y0)
    pro_x_y1_mat = numpy.mat(pro_x_y1)
    pro_x_mat = numpy.multiply(pro_x_y0_mat, pro_y0_mat) + numpy.multiply(pro_x_y1_mat, pro_y1_mat)
    pro_y0_x = numpy.multiply(pro_x_y0_mat, pro_y0_mat) / pro_x_mat
    pro_y1_x = numpy.multiply(pro_x_y1_mat, pro_y1_mat) / pro_x_mat
    return numpy.vstack((pro_y0_x.T, pro_y1_x.T)).T.tolist()  # list m*2


def compute_parameters(x, y, dim_dict):
    '''
    :param x: the training samples (list m*?)
    :param y: the label of training samples (list m*1)
    :param dim_dict: the dimension of dictionary
    :return: the parameters( phi (R), phis0 (list 1*n), phis0 (list 1*n) ) of the model
    '''
    m, k = len(x), dim_dict  # the number of samples , the dimension of dictionary
    num_y0, num_y1 = 0, 0  # statistic the number of y=0, y=1
    num_dict = [[], []]  # statistic the number of x=i|y=0, x=i|y=1
    for i in range(k):
        num_dict[0].append(0)
        num_dict[1].append(0)
    for i in range(m):
        if y[i][0] == 0:
            num_y0 = num_y0 + 1
        else:
            num_y1 = num_y1 + 1
    for i in range(m):
        for ind in x[i]:
            num_dict[y[i][0]][ind] = num_dict[y[i][0]][ind] + 1
    phi = num_y1 / m  # compute phi (R)
    phis0 = []  # compute mu0 (list 1*n) with laplace smoothing
    phis1 = []  # compute mu1 (list 1*n) with laplace smoothing
    for i in range(k):
        phis0_i = (num_dict[0][i] + 1) / (num_y0 + k)
        phis1_i = (num_dict[1][i] + 1) / (num_y1 + k)
        phis0.append(phis0_i)
        phis1.append(phis1_i)
    return phi, phis0, phis1
