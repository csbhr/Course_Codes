import numpy

def gradient_descent(theta, x, y, x_a, y_a, alpha, thre, cost_func, gradient_func, verify_func):
    '''
    update parameters by gradient descent algorithm
    :param theta: the parameters
    :param x: the training samples
    :param y: the ladel of training samples
    :param x_a: the verifying samples
    :param y_a: the labels of verifying samples
    :param alpha: the update rate
    :param thre: the threshold value
    :param cost_func: the cost function
    :param gradient_func: the gradient function (n*c)
    :param verify_func: the verify function
    :return: new parameters , parameters list , cost list, accuracy list
    '''
    theta_list = []  # the parmeters of the training process
    cost_list = []  # the cost of the training process
    accuracy_list = []  # the accuracy of the training process
    t = 0  # update times
    now_cost = cost_func(theta, x, y)  # now value of cost function
    now_accuracy = verify_func(theta, x_a, y_a)  # now value of accuracy
    print("- time:", t, "cost:", now_cost, "accuracy:", now_accuracy)
    theta_list.append(theta)
    cost_list.append(now_cost)
    accuracy_list.append(now_accuracy)
    while 1:
        t += 1
        theta = theta - alpha * gradient_func(theta, x, y)  # update the parameters
        new_cost = cost_func(theta, x, y)
        now_accuracy = verify_func(theta, x_a, y_a)
        print("- time:", t, "cost:", new_cost, "accuracy:", now_accuracy)
        if numpy.abs(new_cost - now_cost) < thre:
            break
        now_cost = new_cost
        theta_list.append(theta)
        cost_list.append(now_cost)
        accuracy_list.append(now_accuracy)
    return theta, theta_list, cost_list, accuracy_list
