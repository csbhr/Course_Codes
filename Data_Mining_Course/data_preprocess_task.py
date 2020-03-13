import tools.file_tools as ftools
import tools.distance_tools as dtools
import os
import numpy as np
import data_preprocess.pca as pca
import data_preprocess.data_preprocess as data_preprocess
import data_preprocess.data_preprocess_20_newsgroup as dp20n
import data_preprocess.data_preprocess_AAAI as dpAAAI
import data_preprocess.data_preprocess_NYSK as dpNYSK

######### 20 newsgroup  ##################################################################################

# root_path = "D:/WorkSpace/python/project_data/Data_Mining/20 Newsgroups/"
# root_path = "C:/WorkSpace/python/python_data/Data_Mining/20 Newsgroups/"

# 读取并存储 20 newsgroup 数据集的字典
# print("读取并存储 20 newsgroup 数据集的字典")
# dictionary = dp20n.read_20_newsgroup_dictionary(root_path)
# ftools.write_txt(root_path + "dictionary_0.txt", dictionary)
# print(len(dictionary))

# 计算并存储 20 newsgroup 的特征向量
# print("计算并存储 20 newsgroup 的特征向量")
# feather_vector = dp20n.read_20_newsgroup_feature_vector(root_path)
# ftools.write_csv(root_path + "feather_vector.csv", feather_vector)

# 计算并存储 20 newsgroup 的权重
# print("计算并存储 20 newsgroup 的权重")
# weights = dp20n.read_20_newsgroup_weight(root_path)
# ftools.write_csv(root_path + "weights.csv", weights)

# 对 20 newsgroup 的特征向量进行pca
# print("对 20 newsgroup 的特征向量进行pca")
# pca_info, feature_vector_after_pca = dp20n.action_20_newsgroup_pca(root_path, "feature_vector.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_feature_vector.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "feature_vector_after_pca.csv", feature_vector_after_pca)
# print("---存储完成--------------------------")

# 对 20 newsgroup 的权重进行pca
# print("对 20 newsgroup 的权重进行pca")
# pca_info, weights_after_pca = dp20n.action_20_newsgroup_pca(root_path, "weights.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_weights.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "weights_after_pca.csv", weights_after_pca)
# print("---存储完成--------------------------")


# 存储 20 newsgroup 的样本之间的交叉距离矩阵
# print("存储 20 newsgroup 的样本之间的交叉距离矩阵")
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_feature_vector.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_feature_vector.csv",
#                                         cal_distance=dtools.cos_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_weights.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_weights.csv",
#                                         cal_distance=dtools.cos_distance)

######### AAAI  ##################################################################################

# root_path = "D:/WorkSpace/python/project_data/Data_Mining/AAAI-14 Accepted Papers - Papers/"
# root_path = "C:/WorkSpace/python/python_data/Data_Mining/AAAI-14 Accepted Papers - Papers/"

# 读取并存储 AAAI 数据集的字典
# print("读取并存储 AAAI 数据集的字典")
# dictionary = dpAAAI.read_AAAI_dictionary(root_path)
# ftools.write_txt(root_path + "dictionary_0.txt", dictionary)
# print(len(dictionary))

# 计算并存储 AAAI 的特征向量
# print("计算并存储 AAAI 的特征向量")
# feature_vector = dpAAAI.read_AAAI_feature_vector(root_path)
# ftools.write_csv(root_path + "feature_vector.csv", feature_vector)

# 计算并存储 AAAI 的权重
# print("计算并存储 AAAI 的权重")
# weights = dpAAAI.read_AAAI_weight(root_path)
# ftools.write_csv(root_path + "weights.csv", weights)

# 对 AAAI 的特征向量进行pca
# print("对 AAAI 的特征向量进行pca")
# pca_info, feature_vector_after_pca = dpAAAI.action_AAAI_pca(root_path, "feature_vector.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_feature_vector.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "feature_vector_after_pca.csv", feature_vector_after_pca)
# print("---存储完成--------------------------")

# 对 AAAI 的权重进行pca
# print("对 AAAI 的权重进行pca")
# pca_info, weights_after_pca = dpAAAI.action_AAAI_pca(root_path, "weights.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_weights.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "weights_after_pca.csv", weights_after_pca)
# print("---存储完成--------------------------")

# 存储 AAAI 的样本之间的交叉距离矩阵
# print("存储 AAAI 的样本之间的交叉距离矩阵")
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_feature_vector.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_feature_vector.csv",
#                                         cal_distance=dtools.cos_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_weights.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_weights.csv",
#                                         cal_distance=dtools.cos_distance)


######### NYSK  ##################################################################################

# root_path = "D:/WorkSpace/python/project_data/Data_Mining/NYSK Data Set/"
# root_path = "C:/WorkSpace/python/python_data/Data_Mining/NYSK Data Set/"

# 读取并存储 NYSK 数据集的字典
# print("读取并存储 NYSK 数据集的字典")
# dictionary = dpNYSK.read_NYSK_dictionary(root_path)
# ftools.write_txt(root_path + "dictionary_summary_0.txt", dictionary)
# print(len(dictionary))

# 计算并存储 NYSK 的特征向量
# print("计算并存储 NYSK 的特征向量")
# feature_vector = dpNYSK.read_NYSK_feature_vector(root_path)
# ftools.write_csv(root_path + "feature_vector.csv", feature_vector)

# 计算并存储 NYSK 的权重
# print("计算并存储 NYSK 的权重")
# weights = dpNYSK.read_NYSK_weight(root_path)
# ftools.write_csv(root_path + "weights.csv", weights)

# 对 NYSK 的特征向量进行pca
# print("对 NYSK 的特征向量进行pca")
# pca_info, feature_vector_after_pca = dpNYSK.action_NYSK_pca(root_path, "feature_vector.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_feature_vector.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "feature_vector_after_pca.csv", feature_vector_after_pca)
# print("---存储完成--------------------------")

# 对 NYSK 的权重进行pca
# print("对 NYSK 的权重进行pca")
# pca_info, weights_after_pca = dpNYSK.action_NYSK_pca(root_path, "weights.csv")
# print("---存储降维矩阵--------------------")
# ftools.pickle_dump(root_path + "pca_weights.data", pca_info)
# print("---存储完成--------------------------")
# print("---存储降维后向量--------------------")
# ftools.write_csv(root_path + "weights_after_pca.csv", weights_after_pca)
# print("---存储完成--------------------------")

# 存储 NYSK 的样本之间的交叉距离矩阵
# print("存储 NYSK 的样本之间的交叉距离矩阵")
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_feature_vector.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "feature_vector_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_feature_vector.csv",
#                                         cal_distance=dtools.cos_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_euclidean_distance_weights.csv",
#                                         cal_distance=dtools.euclidean_distance)
# data_preprocess.get_data_cross_distance(ori_filename=root_path + "weights_after_pca.csv",
#                                         aim_filename=root_path+"cross_cos_distance_weights.csv",
#                                         cal_distance=dtools.cos_distance)
