import numpy
import random


def gradient_descent(theta, x, y, alpha, thre, cost_func, gradient_func):
    '''
    update parameters by gradient descent algorithm
    :param theta: the parameters
    :param x: the training samples
    :param y: the ladel of training samples
    :param alpha: the update rate
    :param thre: the threshold value
    :param cost_func: the cost function
    :param gradient_func: the gradient function (n*c)
    :return: new parameters , parameters list , cost list
    '''
    theta_list = []  # the parmeters of the training process
    cost_list = []  # the cost of the training process
    t = 0  # update times
    now_cost = cost_func(theta, x, y)  # now value of cost function
    print(t, now_cost)
    theta_list.append(theta)
    cost_list.append(now_cost)
    while 1:
        t += 1
        theta = theta - alpha * gradient_func(theta, x, y)  # update the parameters
        new_cost = cost_func(theta, x, y)
        print(t, new_cost)
        if numpy.abs(new_cost - now_cost) < thre:
            break
        now_cost = new_cost
        theta_list.append(theta)
        cost_list.append(now_cost)
    return theta, theta_list, cost_list


def stochastic_gradient_descent(theta, x, y, alpha, times, num_r, cost_func, gradient_func):
    '''
    update parameters by stochastic gradient descent algorithm
    :param theta: the parameters
    :param x: the training samples
    :param y: the ladel of training samples
    :param alpha: the update rate
    :param times: the iteration times
    :param num_r: the number of samples we choose randomly each time
    :param cost_func: the cost function
    :param gradient_func: the gradient function (n*c)
    :return: new parameters , parameters list , cost list
    '''
    theta_list = []  # the parmeters of the training process
    cost_list = []  # the cost of the training process
    m, n = x.shape  # the number of examples,parameters
    t = 0  # update times
    now_cost = cost_func(theta, x, y)  # now value of cost function
    print(t, now_cost)
    best_theta = theta  # save the best theta
    min_cost = now_cost  # save the min cost
    theta_list.append(best_theta)
    cost_list.append(now_cost)
    while t < times:
        t += 1
        i_random = random.sample(range(0, m), num_r)
        x_r = []
        y_r = []
        x_ = x.tolist()
        y_ = y.T.tolist()[0]
        for i in i_random:
            x_r.append(x_[i])
            y_r.append(y_[i])
        x_r = numpy.mat(x_r)
        y_r = numpy.mat(y_r).T
        theta = theta - alpha * gradient_func(theta, x_r, y_r)  # update the parameter
        now_cost = cost_func(theta, x, y)
        print(t, now_cost)
        if now_cost < min_cost:
            best_theta = theta
            min_cost = now_cost
            theta_list.append(best_theta)
        cost_list.append(now_cost)
    return best_theta, min_cost, theta_list, cost_list
