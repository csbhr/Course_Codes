import tools.string_tools as stools
import numpy as np
import tools.file_tools as ftools


def get_txt_terms(text):
    needed_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
    all_terms = stools.cut_word_indicate(text, needed_char)
    return all_terms


def remove_terms_with_number(term_list):
    number_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    remove_list = []
    for term in term_list:
        is_remove = False
        for t in term:
            if t in number_char:
                is_remove = True
                break
        if is_remove:
            remove_list.append(term)
    for re in remove_list:
        term_list.remove(re)
    return term_list


def remove_stopped_terms(term_list):
    term_list = remove_terms_with_number(term_list)
    stopped_terms = ftools.read_txt("stopped_terms.txt")
    used_terms = ftools.read_txt("used_terms.txt")
    remove_list = []
    for term in term_list:
        if term.lower() in stopped_terms or len(term) < 5 or not term.lower() in used_terms:
            remove_list.append(term)
    for re in remove_list:
        term_list.remove(re)
    term_list = [te.lower() for te in term_list]
    term_list = list(set(term_list))
    return term_list


def remove_less_frequency_terms(text_term_list, dictionary, frequency):
    if frequency <= 0:
        return dictionary
    all_include_term = np.zeros([len(dictionary)])  # the times of term appearing in all text
    for i in range(len(dictionary)):
        term = dictionary[i]
        for j in range(len(text_term_list)):
            all_include_term[i] = all_include_term[i] + text_term_list[j].count(term)
    remove_list = []
    for i in range(len(dictionary)):
        if all_include_term[i] <= frequency:
            remove_list.append(dictionary[i])
    for re in remove_list:
        dictionary.remove(re)
    return dictionary


def get_txt_feature_vector(text_term_list, dictionary):
    num_text = len(text_term_list)  # the number of texts
    num_term = len(dictionary)  # the number of terms
    feather_vector = np.zeros([num_text, num_term])  # the weight
    for i in range(num_term):
        term = dictionary[i]
        for j in range(num_text):
            feather_vector[j, i] = text_term_list[j].count(term)
    return feather_vector


def get_txt_weight(text_term_list, dictionary):
    num_text = len(text_term_list)  # the number of texts
    num_term = len(dictionary)  # the number of terms
    text_term_times = np.zeros([num_term, num_text]).tolist()  # the times of term appearing in text
    text_include_term = np.zeros([num_term])  # the number of texts include term
    all_include_term = np.zeros([num_term])  # the times of term appearing in all text
    weights = np.zeros([num_text, num_term])  # the weight
    for i in range(num_term):
        term = dictionary[i]
        for j in range(num_text):
            text_term_times[i][j] = text_term_list[j].count(term)
            all_include_term[i] = all_include_term[i] + text_term_list[j].count(term)
            if term in text_term_list[j]:
                text_include_term[i] = text_include_term[i] + 1
    for i in range(num_text):
        for j in range(num_term):
            weights[i][j] = text_term_times[j][i] * np.log(num_text / text_include_term[j]) + text_term_times[j][i] * (
                    all_include_term[j] / num_text)
    return weights


def get_data_cross_distance(ori_filename, aim_filename, cal_distance):
    data_mat = ftools.read_csv_mat(ori_filename)
    sample_num = data_mat.shape[0]
    cross_distance = np.zeros([sample_num, sample_num])
    for i in range(sample_num):
        for j in range(sample_num):
            cross_distance[i, j] = cal_distance(data_mat[i, :], data_mat[j, :])
    ftools.write_csv(aim_filename, cross_distance.tolist())
