import tools.file_tools as ftools
import data_preprocess.data_preprocess as data_preprocess
import numpy as np
import data_preprocess.pca as pca


def read_NYSK_file_text_term_list(root_path):
    text_data = ftools.read_xml_list_dictionary(root_path+"nysk.xml")
    text_term_list = []
    for one_object in text_data:
        line_content = str(one_object["summary"])
        # line_content = str(one_object["text"])
        line_terms = data_preprocess.get_txt_terms(line_content)
        line_terms = list(set(line_terms))
        line_terms = data_preprocess.remove_stopped_terms(line_terms)
        text_term_list.append(line_terms)
    return text_term_list


def read_NYSK_dictionary(root_path):
    text_term_list = read_NYSK_file_text_term_list(root_path)
    dictionary = []
    for text_term in text_term_list:
        dictionary.extend(text_term)
    dictionary = list(set(dictionary))
    dictionary = data_preprocess.remove_less_frequency_terms(text_term_list, dictionary, frequency=0)
    return dictionary


def read_NYSK_feature_vector(root_path):
    text_term_list = read_NYSK_file_text_term_list(root_path)
    dictionary = ftools.read_txt(root_path + "dictionary.txt")
    feature_vector = data_preprocess.get_txt_feature_vector(text_term_list, dictionary)
    return feature_vector


def read_NYSK_weight(root_path):
    text_term_list = read_NYSK_file_text_term_list(root_path)
    dictionary = ftools.read_txt(root_path + "dictionary.txt")
    weights = data_preprocess.get_txt_weight(text_term_list, dictionary)
    return weights


def action_NYSK_pca(root_path, origin_filename):
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
