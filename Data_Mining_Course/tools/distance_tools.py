import numpy as np


def euclidean_distance(vector_1, vector_2):
    '''
    calculate euclidean distance
    :param vector_1: numpy.mat (1*n)
    :param vector_2: numpy.mat (1*n)
    :return: float
    '''
    return float((np.sqrt((vector_1 - vector_2) * (vector_1 - vector_2).T)))


def cos_distance(vector_1, vector_2):
    '''
    calculate cos distance
    :param vector_1: numpy.mat (1*n)
    :param vector_2: numpy.mat (1*n)
    :return: 1-cos(vector_1, vector_2) float
    '''
    sum_1 = float(vector_1 * vector_2.T)
    sum_2 = float(np.sqrt(vector_1 * vector_1.T))
    sum_3 = float(np.sqrt(vector_2 * vector_2.T))
    return 1 - sum_1 / (sum_2 * sum_3)
