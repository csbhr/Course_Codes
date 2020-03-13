import numpy
import needless.GD_5flod as gd
import tools.ml_sample_tools as mstools
import matplotlib_helper.figure as figure


############################### logistic regression begin ###############################

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


def verify_function(theta, x, y):
    '''
    :param theta: the parameters
    :param x: the verifying samples
    :param y: the labels of verifying samples
    :return: verify the accuracy with the verifying samples set
    '''
    m = x.shape[0]  # the number of verifying samples
    y_ = h(theta, x)
    y_ = numpy.argmax(y_, axis=1)
    diff = numpy.abs(y - y_)
    correct_num = 0
    for i in range(m):
        if diff[i] < 0.5:
            correct_num = correct_num + 1
    return float(correct_num) / float(m)


def train(theta, x, y, x_a, y_a, alpha, thre):
    '''
    update parameters by GD
    :param theta: the parameters
    :param x: the training samples
    :param y: the labels of training samples
    :param x_a: the verifying samples
    :param y_a: the labels of verifying samples
    :param alpha: the update rate
    :param thre: the threshold value
    :return: new parameters , parameters list , cost list, accuracy list
    '''
    return gd.gradient_descent(theta, x, y, x_a, y_a, alpha, thre, cost_func=cost_function,
                               gradient_func=gradient_function, verify_func=verify_function)


############################### logistic regression end   ###############################

def get_data(path):
    data = numpy.loadtxt(path)
    return data


def action_basic_one(theta, sample_train, sample_verify):
    return train(theta, sample_train[0], sample_train[1], sample_verify[0], sample_verify[1], alpha=0.000015,
                 thre=0.00001)


def action_basic(theta, groups):
    messages = list([])
    print("Performance 1(training set:group 1,2,3,4 ; verify set:group 5)")
    theta, t_list, c_list, a_list = action_basic_one(theta, [
        numpy.vstack((groups[0][0], groups[1][0], groups[2][0], groups[3][0])),
        numpy.vstack((groups[0][1], groups[1][1], groups[2][1], groups[3][1]))], [groups[4][0], groups[4][1]])
    messages.append([theta, "training set:group 1,2,3,4 ; verify set:group 5", t_list, c_list, a_list])
    print("Training Done!")
    print("Performance 2(training set:group 1,2,3,5 ; verify set:group 4)")
    theta, t_list, c_list, a_list = action_basic_one(theta, [
        numpy.vstack((groups[0][0], groups[1][0], groups[2][0], groups[4][0])),
        numpy.vstack((groups[0][1], groups[1][1], groups[2][1], groups[4][1]))], [groups[3][0], groups[3][1]])
    messages.append([theta, "training set:group 1,2,3,5 ; verify set:group 4", t_list, c_list, a_list])
    print("Training Done!")
    print("Performance 3(training set:group 1,2,4,5 ; verify set:group 3)")
    theta, t_list, c_list, a_list = action_basic_one(theta, [
        numpy.vstack((groups[0][0], groups[1][0], groups[3][0], groups[4][0])),
        numpy.vstack((groups[0][1], groups[1][1], groups[3][1], groups[4][1]))], [groups[2][0], groups[2][1]])
    messages.append([theta, "training set:group 1,2,4,5 ; verify set:group 3", t_list, c_list, a_list])
    print("Training Done!")
    print("Performance 4(training set:group 1,3,4,5 ; verify set:group 2)")
    theta, t_list, c_list, a_list = action_basic_one(theta, [
        numpy.vstack((groups[0][0], groups[2][0], groups[3][0], groups[4][0])),
        numpy.vstack((groups[0][1], groups[2][1], groups[3][1], groups[4][1]))], [groups[1][0], groups[1][1]])
    messages.append([theta, "training set:group 1,3,4,5 ; verify set:group 2", t_list, c_list, a_list])
    print("Training Done!")
    print("Performance 5(training set:group 2,3,4,5 ; verify set:group 1)")
    theta, t_list, c_list, a_list = action_basic_one(theta, [
        numpy.vstack((groups[1][0], groups[2][0], groups[3][0], groups[4][0])),
        numpy.vstack((groups[1][1], groups[2][1], groups[3][1], groups[4][1]))], [groups[0][0], groups[0][1]])
    messages.append([theta, "training set:group 2,3,4,5 ; verify set:group 1", t_list, c_list, a_list])
    print("Training Done!")
    return messages


def action():
    x_ = get_data("../data_files/ex4x.dat")
    y = get_data("../data_files/ex4y.dat")
    x = []
    for i in range(len(x_)):
        x.append([1, x_[i][0], x_[i][1]])
    theta = [[0., 0.],
             [0., 0.],
             [0., 0.]]
    x = numpy.mat(x)  # turn x,y,theta to mat
    y = numpy.mat(y).T
    groups = mstools.group([x, y], 5)
    theta = numpy.mat(theta)
    messages=action_basic(theta,groups)
    max = messages[0]
    for i in range(5):
        if (messages[i][4][-1] >= max[4][-1]):
            max = messages[i]
    print("The best of the 5 performaces:", max[1])
    print("* The parameters of hidden hidden layer:")
    print(max[0])
    print("* The accuracy:", max[4][-1])
    print("* The iteration times:", len(max[3]))
    figure.pic_5([messages[0][4][-1], messages[1][4][-1], messages[2][4][-1], messages[3][4][-1], messages[4][4][-1]],
                 [messages[0][3][-1], messages[1][3][-1], messages[2][3][-1], messages[3][3][-1], messages[4][3][-1]],
                 [len(messages[0][3]), len(messages[1][3]), len(messages[2][3]), len(messages[3][3]),
                  len(messages[4][3])])

###############################     project begin         ###############################

###############################     project end           ###############################

action()
