import tools.file_tools as ftools
import data_preprocess.data_preprocess as data_preprocess
import os
import numpy as np
import data_preprocess.pca as pca


def read_20_newsgroup_file_term(text_path):
    file_txt_lines = ftools.read_txt(text_path)
    file_txt_lines_terms = [data_preprocess.get_txt_terms(line) for line in file_txt_lines]
    lines_num = -1
    for line_term in file_txt_lines_terms:
        if len(line_term) > 1 and line_term[0] == "Lines":
            lines_num = int(line_term[1])
            break
    needed_lines_terms = file_txt_lines_terms[-lines_num:]
    file_terms = []
    for nlt in needed_lines_terms:
        file_terms.extend(nlt)
    file_terms = list(set(file_terms))
    file_terms = data_preprocess.remove_stopped_terms(file_terms)
    return file_terms


def read_20_newsgroup_dictionary(root_path):
    file_path_list = []
    dictionary = []  # the dictionary
    text_term_list = []  # the list of text terms
    sub_dir = os.listdir(root_path + "mini_newsgroups/")
    for sd in sub_dir:
        file_names = os.listdir(root_path + "mini_newsgroups/" + sd + "/")
        for fn in file_names:
            file_path_list.append(root_path + "mini_newsgroups/" + sd + "/" + fn)
    for fp in file_path_list:
        text_term_list.append(read_20_newsgroup_file_term(fp))
        dictionary.extend(read_20_newsgroup_file_term(fp))
    dictionary = list(set(dictionary))
    dictionary = data_preprocess.remove_less_frequency_terms(text_term_list, dictionary, frequency=0)
    return dictionary


def read_20_newsgroup_feature_vector(root_path):
    file_path_list = []
    sub_dir = os.listdir(root_path + "mini_newsgroups/")
    text_term_list = []
    dictionary = ftools.read_txt(root_path + "dictionary.txt")
    for sd in sub_dir:
        file_names = os.listdir(root_path + "mini_newsgroups/" + sd + "/")
        for fn in file_names:
            file_path_list.append(root_path + "mini_newsgroups/" + sd + "/" + fn)
    for fp in file_path_list:
        text_term_list.append(read_20_newsgroup_file_term(fp))
    feature_vector = data_preprocess.get_txt_feature_vector(text_term_list, dictionary)
    return feature_vector


def read_20_newsgroup_weight(root_path):
    file_path_list = []
    sub_dir = os.listdir(root_path + "mini_newsgroups/")
    text_term_list = []
    dictionary = ftools.read_txt(root_path + "dictionary.txt")
    for sd in sub_dir:
        file_names = os.listdir(root_path + "mini_newsgroups/" + sd + "/")
        for fn in file_names:
            file_path_list.append(root_path + "mini_newsgroups/" + sd + "/" + fn)
    for fp in file_path_list:
        text_term_list.append(read_20_newsgroup_file_term(fp))
    weights = data_preprocess.get_txt_weight(text_term_list, dictionary)
    return weights


def action_20_newsgroup_pca(root_path, origin_filename):
    print("---正在读取输入矩阵------------------")
    origin_data = ftools.read_csv_list(root_path + origin_filename)
    print("---读取输入矩阵完成------------------")
    origin_data_mat = np.mat(origin_data)
    print("---计算特征值、特征向量--------------")
    e, EV, mean = pca.pca_by_retention(origin_data_mat, 0.7)
    # e, EV, mean = pca.pca_by_k(origin_data_mat, 20)
    print("---计算完成--------------------------")
    data_after_pca = []
    print("---降维中----------------------------")
    for od in origin_data:
        od_mat = np.mat(od)
        data_after_pca.append(pca.reduce_dimensionality_vector(od_mat, [e, EV, mean]).tolist()[0])
    print("---降维完成--------------------------")
    return [e, EV, mean], data_after_pca
