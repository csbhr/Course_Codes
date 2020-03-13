import numpy


def sigmoid_function(z):
    '''
    :param z: the input
    :return: the value of sigmoid function
    '''
    return 1.0 / (1.0 + numpy.exp(-z))


def all_conn_compute(weight, bias, x):
    '''
    :param weight: the weight
    :param bias: the bias
    :param x: the input value
    :return: all connection value with sigmoid function as active function
    '''
    return sigmoid_function(x * weight + bias)


def hidden_layer(weight, bias, x):
    '''
    :param weight: the weight of hidden layer
    :param bias: the bias of hidden layer
    :param x: the input of hidden layer
    :return: the output of of hidden layer
    '''
    return all_conn_compute(weight, bias, x)


def output_layer(weight, bias, b):
    '''
    :param weight: the weight of output layer
    :param bias: the bias of output layer
    :param b: the input of output layer
    :return: the output of of output layer
    '''
    return all_conn_compute(weight, bias, b)


def cost_function(w_h, b_h, w_o, b_o, x, y):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :return: the value of cost fucntion (m*1)
    '''
    b = hidden_layer(w_h, b_h, x)
    h = output_layer(w_o, b_o, b)
    return numpy.sum(numpy.multiply((h - y), (h - y)))


def error_outputlayer(w_h, b_h, w_o, b_o, x, y):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :return: the value of error^outputlayer (m*l)
    '''
    b = hidden_layer(w_h, b_h, x)
    h = output_layer(w_o, b_o, b)
    return numpy.multiply(numpy.multiply((h - y), h), (1 - h))


def error_hiddenlayer(w_h, b_h, w_o, b_o, x, y):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :return: the value of error^hiddenlayer (m*q)
    '''
    b = hidden_layer(w_h, b_h, x)
    error_o = error_outputlayer(w_h, b_h, w_o, b_o, x, y)
    return numpy.multiply(numpy.multiply(error_o * w_o.T, b), (1 - b))


def gradient_function(w_h, b_h, w_o, b_o, x, y):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :return: the gradient value of w_h,b_h,w_o,b_o
    '''
    b = hidden_layer(w_h, b_h, x)
    error_o = error_outputlayer(w_h, b_h, w_o, b_o, x, y)
    error_h = error_hiddenlayer(w_h, b_h, w_o, b_o, x, y)
    grad_w_o = b.T * error_o  # the gradient of w_o
    one_vector = numpy.ones([x.shape[0], 1])  # a m*1 vector with all items is one
    grad_b_o = one_vector.T * error_o  # the gradient of b_o
    grad_w_h = x.T * error_h  # the gradient of w_h
    grad_b_h = one_vector.T * error_h  # the gradient of b_h
    return grad_w_h, grad_b_h, grad_w_o, grad_b_o


def verify(w_h, b_h, w_o, b_o, x, y):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the verifying samples
    :param y: the label of verifying samples
    :return: verify the accuracy with the verifying samples set
    '''
    m = x.shape[0]  # the number of verifying samples
    b = hidden_layer(w_h, b_h, x)
    h = output_layer(w_o, b_o, b)
    diff = numpy.sum(numpy.abs(y - h), axis=1)
    correct_num = 0
    for i in range(m):
        if (diff[i] < 0.5):
            correct_num = correct_num + 1
    return float(correct_num) / float(m)


def train_with_one(w_h, b_h, w_o, b_o, x, y, alpha):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :param alpha: the training rate
    :return: the new w_h,b_h,w_o,b_o after one trip train with only one sample
    '''
    m = x.shape[0]  # the number of training samples
    for i in range(m):
        x_i = x[i]
        y_i = y[i]
        grad_w_h, grad_b_h, grad_w_o, grad_b_o = gradient_function(w_h, b_h, w_o, b_o, x_i, y_i)
        w_h = w_h - alpha * grad_w_h
        b_h = b_h - alpha * grad_b_h
        w_o = w_o - alpha * grad_w_o
        b_o = b_o - alpha * grad_b_o
    return w_h, b_h, w_o, b_o


def train_with_all(w_h, b_h, w_o, b_o, x, y, alpha):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :param alpha: the training rate
    :return: the new w_h,b_h,w_o,b_o after one trip train with all samples
    '''
    grad_w_h, grad_b_h, grad_w_o, grad_b_o = gradient_function(w_h, b_h, w_o, b_o, x, y)
    w_h = w_h - alpha * grad_w_h
    b_h = b_h - alpha * grad_b_h
    w_o = w_o - alpha * grad_w_o
    b_o = b_o - alpha * grad_b_o
    return w_h, b_h, w_o, b_o


def train(w_h, b_h, w_o, b_o, x, y, x_a, y_a, alpha, thre):
    '''
    :param w_h: the weight of hidden layer
    :param b_h: the bias of hidden layer
    :param w_o: the weight of output layer
    :param b_o: the input of output layer
    :param x: the training samples
    :param y: the label of training samples
    :param x_a: the verifying samples
    :param y_a: the label of verifying samples
    :param alpha: the training rate
    :param thre: the threshold value
    :return: the new w_h,b_h,w_o,b_o after train ,and parameter_list, cost_list, accuracy_list
    '''
    parameter_list = []  # the parmeters of the training process
    cost_list = []  # the cost of the training process
    accuracy_list = []  # the accuracy of the training process
    t = 0
    now_cost = cost_function(w_h, b_h, w_o, b_o, x, y)  # now value of cost function
    now_accuracy = verify(w_h, b_h, w_o, b_o, x_a, y_a)  # now value of accuracy
    print("- time:", t, "cost:", now_cost, "accuracy:", now_accuracy)
    parameter_list.append([w_h, b_h, w_o, b_o])
    cost_list.append(now_cost)
    accuracy_list.append(now_accuracy)
    while 1:
        t = t + 1
        # w_h, b_h, w_o, b_o = train_with_one(w_h, b_h, w_o, b_o, x, y, alpha)    # once train with only one sample
        w_h, b_h, w_o, b_o = train_with_all(w_h, b_h, w_o, b_o, x, y, alpha)  # once train with all samples
        new_cost = cost_function(w_h, b_h, w_o, b_o, x, y)  # now value of cost function
        now_accuracy = verify(w_h, b_h, w_o, b_o, x_a, y_a)
        print("- time:", t, "cost:", new_cost, "accuracy:", now_accuracy)
        if numpy.abs(new_cost - now_cost) < thre:
            break
        now_cost = new_cost
        parameter_list.append([w_h, b_h, w_o, b_o])
        cost_list.append(now_cost)
        accuracy_list.append(now_accuracy)
    return w_h, b_h, w_o, b_o, parameter_list, cost_list, accuracy_list
