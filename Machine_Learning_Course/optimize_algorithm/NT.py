def newton_method(theta, x, y, times, cost_func, gradient_func, hessian_function):
    '''
    update parameters by newton's method
    :param theta: the parameters
    :param x: the training samples
    :param y: the ladel of training samples
    :param times: the iteration times
    :param cost_func: the cost function
    :param gradient_func: the gradient function (n*c)
    :param hessian_function: the hessian mat of the cost function (n*n)
    :return: new parameters , parameters list , cost list
    '''
    theta_list = []  # the parmeters of the training process
    cost_list = []  # the cost of the training process
    t = 0  # update times
    now_cost = cost_func(theta, x, y)  # now value of cost function
    print(t, now_cost)
    theta_list.append(theta)
    cost_list.append(now_cost)
    while t < times:
        t += 1
        grad = gradient_func(theta, x, y)  # gradient
        hess = hessian_function(theta, x)  # Hessian mat
        theta = theta - hess.I * grad  # update the parameter
        now_cost = cost_func(theta, x, y)
        print(t, now_cost)
        theta_list.append(theta)
        cost_list.append(now_cost)
    return theta, theta_list, cost_list
