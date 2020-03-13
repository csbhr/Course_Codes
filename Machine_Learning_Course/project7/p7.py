import tools.file_tools as ftools
import naive_bayes.naive_bayes_classifier as nbc
import naive_bayes.naive_bayes_MBEM as nbMBEM
import matplotlib_helper.figure as figure
import numpy


def write_data_tofile():
    x_train_1 = [[1, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 1, 0, 0],
                 [0, 1, 1, 0, 0, 1]]
    x_train_2 = [[1, 0, 1],
                 [1, 1, 4],
                 [1, 3],
                 [5, 2, 1]]
    y_train = [[0],
               [0],
               [0],
               [1]]
    x_verify_1 = [[0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 1]]
    x_verify_2 = [[1, 1, 1, 5, 2],
                  [5, 5, 2, 4]]
    y_verify = [[1],
                [0]]
    ftools.pickle_dump("data_files/naive_bayes_classifier_train.dat", [x_train_1, y_train])
    ftools.pickle_dump("data_files/naive_bayes_classifier_verify.dat", [x_verify_1, y_verify])
    ftools.pickle_dump("data_files/naive_bayes_MBEM_train.dat", [x_train_2, y_train])
    ftools.pickle_dump("data_files/naive_bayes_MBEM_verify.dat", [x_verify_2, y_verify])


def action_nbc():
    train_samples = ftools.pickle_load("data_files/naive_bayes_classifier_train.dat")
    verify_samples = ftools.pickle_load("data_files/naive_bayes_classifier_verify.dat")
    x = numpy.mat(train_samples[0])
    y = numpy.mat(train_samples[1])
    x_verify = numpy.mat(verify_samples[0])
    phi, phis0, phis1 = nbc.compute_parameters(x, y)
    predict_verify = nbc.h(phi, phis0, phis1, x_verify).tolist()
    for i in range(len(predict_verify)):
        print("Sample ", i, ": ", "P(y=0|x)=", predict_verify[i][0], "   ", "P(y=1|x)=", predict_verify[i][1])
    figure.pic_7(phis0, phis1, predict_verify)


def action_nbMBEM():
    train_samples = ftools.pickle_load("data_files/naive_bayes_MBEM_train.dat")
    verify_samples = ftools.pickle_load("data_files/naive_bayes_MBEM_verify.dat")
    x = train_samples[0]
    y = train_samples[1]
    x_verify = verify_samples[0]
    phi, phis0, phis1 = nbMBEM.compute_parameters(x, y, 6)
    predict_verify = nbMBEM.h(phi, phis0, phis1, x_verify)
    for i in range(len(predict_verify)):
        print("Sample ", i, ": ", "P(y=0|x)=", predict_verify[i][0], "   ", "P(y=1|x)=", predict_verify[i][1])
    figure.pic_7(phis0, phis1, predict_verify)


def action():
    # action_nbc()
    action_nbMBEM()
