import full_connect_nn.full_connect_nn as fcn
import full_connect_nn.full_connect_nn_tensorflow as fcnf
import matplotlib_helper.figure as figure
import tools.file_tools as ftools
import tools.ml_sample_tools as mstools
import numpy


def get_data(path):
    data = numpy.loadtxt(path)
    return data


def get_model_file(path):
    return ftools.pickle_load(path)


def verify_model(x, y):
    parameters = get_model_file("project5/parameters_nontensorflow.data")
    w_h = parameters[0]
    b_h = parameters[1]
    w_o = parameters[2]
    b_o = parameters[3]
    return fcn.verify(w_h, b_h, w_o, b_o, x, y)


def action_basic_one(w_h, b_h, w_o, b_o, sample_train, sample_verify):
    return fcn.train(w_h, b_h, w_o, b_o, sample_train[0], sample_train[1], sample_verify[0], sample_verify[1],
                     alpha=0.01, thre=0.00001)


def action_tensorflow_one(sample_train, sample_verify):
    return fcnf.train([2, 3, 1], sample_train, sample_verify, alpha=0.01, times=5000)


def action_basic(w_h, b_h, w_o, b_o, x_group, y_group):
    messages = list([])
    print("Performance 1(training set:group 1,2,3,4 ; verify set:group 5)")
    w_h_t, b_h_t, w_o_t, b_o_t, p_list, c_list, a_list = action_basic_one(w_h, b_h, w_o, b_o, [
        numpy.vstack((x_group[0], x_group[1], x_group[2], x_group[3])),
        numpy.vstack((y_group[0], y_group[1], y_group[2], y_group[3]))], [x_group[4], y_group[4]])
    messages.append(
        [w_h_t, b_h_t, w_o_t, b_o_t, a_list[-1], "training set:group 1,2,3,4 ; verify set:group 5", p_list, c_list,
         a_list])
    print("Training Done!")
    print("Performance 2(training set:group 1,2,3,5 ; verify set:group 4)")
    w_h_t, b_h_t, w_o_t, b_o_t, p_list, c_list, a_list = action_basic_one(w_h, b_h, w_o, b_o, [
        numpy.vstack((x_group[0], x_group[1], x_group[2], x_group[4])),
        numpy.vstack((y_group[0], y_group[1], y_group[2], y_group[4]))], [x_group[3], y_group[3]])
    messages.append(
        [w_h_t, b_h_t, w_o_t, b_o_t, a_list[-1], "training set:group 1,2,3,5 ; verify set:group 4", p_list, c_list,
         a_list])
    print("Training Done!")
    print("Performance 3(training set:group 1,2,4,5 ; verify set:group 3)")
    w_h_t, b_h_t, w_o_t, b_o_t, p_list, c_list, a_list = action_basic_one(w_h, b_h, w_o, b_o, [
        numpy.vstack((x_group[0], x_group[1], x_group[3], x_group[4])),
        numpy.vstack((y_group[0], y_group[1], y_group[3], y_group[4]))], [x_group[2], y_group[2]])
    messages.append(
        [w_h_t, b_h_t, w_o_t, b_o_t, a_list[-1], "training set:group 1,2,4,5 ; verify set:group 3", p_list, c_list,
         a_list])
    print("Training Done!")
    print("Performance 4(training set:group 1,3,4,5 ; verify set:group 2)")
    w_h_t, b_h_t, w_o_t, b_o_t, p_list, c_list, a_list = action_basic_one(w_h, b_h, w_o, b_o, [
        numpy.vstack((x_group[0], x_group[2], x_group[3], x_group[4])),
        numpy.vstack((y_group[0], y_group[2], y_group[3], y_group[4]))], [x_group[1], y_group[1]])
    messages.append(
        [w_h_t, b_h_t, w_o_t, b_o_t, a_list[-1], "training set:group 1,3,4,5 ; verify set:group 2", p_list, c_list,
         a_list])
    print("Training Done!")
    print("Performance 5(training set:group 2,3,4,5 ; verify set:group 1)")
    w_h_t, b_h_t, w_o_t, b_o_t, p_list, c_list, a_list = action_basic_one(w_h, b_h, w_o, b_o, [
        numpy.vstack((x_group[1], x_group[2], x_group[3], x_group[4])),
        numpy.vstack((y_group[1], y_group[2], y_group[3], y_group[4]))], [x_group[0], y_group[0]])
    messages.append(
        [w_h_t, b_h_t, w_o_t, b_o_t, a_list[-1], "training set:group 2,3,4,5 ; verify set:group 1", p_list, c_list,
         a_list])
    print("Training Done!")
    return messages


def action_tensorflow(x_group, y_group):
    messages = list([])
    print("Performance 1(training set:group 1,2,3,4 ; verify set:group 5)")
    w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t = action_tensorflow_one([
        numpy.vstack((x_group[0], x_group[1], x_group[2], x_group[3])),
        numpy.vstack((y_group[0], y_group[1], y_group[2], y_group[3]))],
        [x_group[4], y_group[4]])
    messages.append([w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t, "training set:group 1,2,3,4 ; verify set:group 5"])
    print("Training Done!")
    print("Performance 2(training set:group 1,2,3,5 ; verify set:group 4)")
    w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t = action_tensorflow_one([
        numpy.vstack((x_group[0], x_group[1], x_group[2], x_group[4])),
        numpy.vstack((y_group[0], y_group[1], y_group[2], y_group[4]))],
        [x_group[3], y_group[3]])
    messages.append([w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t, "training set:group 1,2,3,5 ; verify set:group 4"])
    print("Training Done!")
    print("Performance 3(training set:group 1,2,4,5 ; verify set:group 3)")
    w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t = action_tensorflow_one([
        numpy.vstack((x_group[0], x_group[1], x_group[3], x_group[4])),
        numpy.vstack((y_group[0], y_group[1], y_group[3], y_group[4]))],
        [x_group[2], y_group[2]])
    messages.append([w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t, "training set:group 1,2,4,5 ; verify set:group 3"])
    print("Training Done!")
    print("Performance 4(training set:group 1,3,4,5 ; verify set:group 2)")
    w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t = action_tensorflow_one([
        numpy.vstack((x_group[0], x_group[2], x_group[3], x_group[4])),
        numpy.vstack((y_group[0], y_group[2], y_group[3], y_group[4]))],
        [x_group[1], y_group[1]])
    messages.append([w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t, "training set:group 1,3,4,5 ; verify set:group 2"])
    print("Training Done!")
    print("Performance 5(training set:group 2,3,4,5 ; verify set:group 1)")
    w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t = action_tensorflow_one([
        numpy.vstack((x_group[1], x_group[2], x_group[3], x_group[4])),
        numpy.vstack((y_group[1], y_group[2], y_group[3], y_group[4]))],
        [x_group[0], y_group[0]])
    messages.append([w_h_t, b_h_t, w_o_t, b_o_t, accuracy_t, "training set:group 2,3,4,5 ; verify set:group 1"])
    print("Training Done!")
    return messages


def action():
    x = get_data("data_files/ex4x.dat")
    y = get_data("data_files/ex4y.dat")
    x = numpy.mat(x) / 100
    y = numpy.mat(y).T
    groups = mstools.group([x, y], 5)
    x_group = []
    y_group = []
    for g in groups:
        x_group.append(g[0])
        y_group.append(g[1])
    w_h = numpy.mat(numpy.random.rand(2, 3))
    b_h = numpy.mat(numpy.random.rand(1, 3))
    w_o = numpy.mat(numpy.random.rand(3, 1))
    b_o = numpy.mat(numpy.random.rand(1, 1))
    messages = action_basic(w_h, b_h, w_o, b_o, x_group, y_group)  # do not use tensorflow
    # messages = action_tensorflow(x_group, y_group)  # use tensorflow
    max = messages[0]
    for i in range(5):
        if messages[i][4] >= max[4]:
            max = messages[i]
    ftools.pickle_dump("project5/parameters_nontensorflow.data",
                       [max[0], max[1], max[2], max[3]])  # dumping the model to file
    print("The best of the 5 performaces:", max[5])
    print("* The weight of hidden hidden layer:")
    print(max[0])
    print("* The bias of hidden hidden layer:")
    print(max[1])
    print("* The weight of hidden output layer:")
    print(max[2])
    print("* The bias of hidden output layer:")
    print(max[3])
    print("* The accuracy:", max[4])
    print("* The iteration times:", len(max[7]))  # do not use tensorflow
    figure.pic_4([messages[0][7], messages[1][7], messages[2][7], messages[3][7], messages[4][7]],
                 [messages[0][8], messages[1][8], messages[2][8], messages[3][8], messages[4][8]],
                 interval=50)  # do not use tensorflow
    figure.pic_5([messages[0][4], messages[1][4], messages[2][4], messages[3][4], messages[4][4]],
                 [messages[0][7][-1], messages[1][7][-1], messages[2][7][-1], messages[3][7][-1], messages[4][7][-1]],
                 [len(messages[0][7]), len(messages[1][7]), len(messages[2][7]), len(messages[3][7]),
                  len(messages[4][7])])  # do not use tensorflow
