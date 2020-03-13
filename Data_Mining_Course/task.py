import text_clustering.k_means as k_means
import text_clustering.dbscan as dbscan
import text_clustering.fcm as fcm
import text_clustering.agnes as agnes
import tools.distance_tools as dtools
import tools.file_tools as ftools
import numpy as np
import time

root_path = "D:/WorkSpace/python/project_data/Data_Mining/"
root_path_result = root_path + "result/"
root_path_20_newsgroup = root_path + "20 Newsgroups/"
root_path_AAAI = root_path + "AAAI-14 Accepted Papers - Papers/"
root_path_NYSK = root_path + "NYSK Data Set/"

########## K-Means ###########################

# for k in range(11)[2:]:
#     kmeans = k_means.K_MEANS(
#         # data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#         data_path=root_path_20_newsgroup + "weights_after_pca.csv",
#         # cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
#         # cross_distance_path=root_path_20_newsgroup + "cross_euclidean_distance_feature_vector.csv",
#         # cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_weights.csv",
#         cross_distance_path=root_path_20_newsgroup + "cross_euclidean_distance_weights.csv",
#         K=k,
#         # cal_distance=dtools.cos_distance)
#         cal_distance=dtools.euclidean_distance)
#     kmeans.cluster()
#     SSE_value = kmeans.get_SSE()
#     coefficient_outline = kmeans.get_coefficient_outline()
#     # ftools.append_txt(root_path_result + "20_Newsgroups_kmeans.txt",
#     #                   [str(k) + ",feature_vector,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "20_Newsgroups_kmeans.txt",
#     #                   [str(k) + ",feature_vector,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "20_Newsgroups_kmeans.txt",
#     #                   [str(k) + ",weights,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     ftools.append_txt(root_path_result + "20_Newsgroups_kmeans.txt",
#                       [str(k) + ",weights,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])

# for k in range(11)[2:]:
#     kmeans = k_means.K_MEANS(
#         # data_path=root_path_AAAI + "feature_vector_after_pca.csv",
#         data_path=root_path_AAAI + "weights_after_pca.csv",
#         # cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
#         # cross_distance_path=root_path_AAAI + "cross_euclidean_distance_feature_vector.csv",
#         # cross_distance_path=root_path_AAAI + "cross_cos_distance_weights.csv",
#         cross_distance_path=root_path_AAAI + "cross_euclidean_distance_weights.csv",
#         K=k,
#         # cal_distance=dtools.cos_distance)
#         cal_distance=dtools.euclidean_distance)
#     kmeans.cluster()
#     SSE_value = kmeans.get_SSE()
#     coefficient_outline = kmeans.get_coefficient_outline()
#     # ftools.append_txt(root_path_result + "AAAI_kmeans.txt",
#     #                   [str(k) + ",feature_vector,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "AAAI_kmeans.txt",
#     #                   [str(k) + ",feature_vector,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "AAAI_kmeans.txt",
#     #                   [str(k) + ",weights,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     ftools.append_txt(root_path_result + "AAAI_kmeans.txt",
#                       [str(k) + ",weights,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])

# for k in range(11)[2:]:
#     kmeans = k_means.K_MEANS(
#         # data_path=root_path_NYSK + "feature_vector_after_pca.csv",
#         data_path=root_path_NYSK + "weights_after_pca.csv",
#         # cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
#         # cross_distance_path=root_path_NYSK + "cross_euclidean_distance_feature_vector.csv",
#         # cross_distance_path=root_path_NYSK + "cross_cos_distance_weights.csv",
#         cross_distance_path=root_path_NYSK + "cross_euclidean_distance_weights.csv",
#         K=k,
#         # cal_distance=dtools.cos_distance)
#         cal_distance=dtools.euclidean_distance)
#     kmeans.cluster()
#     SSE_value = kmeans.get_SSE()
#     coefficient_outline = kmeans.get_coefficient_outline()
#     # ftools.append_txt(root_path_result + "NYSK_kmeans.txt",
#     #                   [str(k) + ",feature_vector,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "NYSK_kmeans.txt",
#     #                   [str(k) + ",feature_vector,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     # ftools.append_txt(root_path_result + "NYSK_kmeans.txt",
#     #                   [str(k) + ",weights,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#     ftools.append_txt(root_path_result + "NYSK_kmeans.txt",
#                       [str(k) + ",weights,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])


# print("20 newsgroups 数据集的 K-means 算法：")
# begin_time = time.time()
# kmeans = k_means.K_MEANS(
#     data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     cal_distance=dtools.cos_distance)
# kmeans.cluster()
# end_time = time.time()
# kmeans.show_cluster_allocation()
# SSE_value = kmeans.get_SSE()
# coefficient_outline = kmeans.get_coefficient_outline()
# print("SSE：", SSE_value)
# print("轮廓系数：", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("AAAI-14 数据集的 K-means 算法：")
# begin_time = time.time()
# kmeans = k_means.K_MEANS(
#     data_path=root_path_AAAI + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
#     K=7,
#     cal_distance=dtools.cos_distance)
# kmeans.cluster()
# end_time = time.time()
# kmeans.show_cluster_allocation()
# SSE_value = kmeans.get_SSE()
# coefficient_outline = kmeans.get_coefficient_outline()
# print("SSE：", SSE_value)
# print("轮廓系数：", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("NYSK 数据集的 K-means 算法：")
# begin_time = time.time()
# kmeans = k_means.K_MEANS(
#     data_path=root_path_NYSK + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     cal_distance=dtools.cos_distance)
# kmeans.cluster()
# end_time = time.time()
# kmeans.show_cluster_allocation()
# SSE_value = kmeans.get_SSE()
# coefficient_outline = kmeans.get_coefficient_outline()
# print("SSE：", SSE_value)
# print("轮廓系数：", coefficient_outline)
# print("用时：", end_time - begin_time, "s")


########## AGNES #########################

# print("20 newsgroups 数据集的 AGNES 算法：")
# begin_time = time.time()
# agnes_ = agnes.AGNES(
#     data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     cal_distance=dtools.cos_distance)
# agnes_.cluster()
# end_time = time.time()
# SSE_value = agnes_.get_SSE()
# coefficient_outline = agnes_.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("AAAI 数据集的 AGNES 算法：")
# begin_time = time.time()
# agnes_ = agnes.AGNES(
#     data_path=root_path_AAAI + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
#     K=7,
#     cal_distance=dtools.cos_distance)
# agnes_.cluster()
# end_time = time.time()
# SSE_value = agnes_.get_SSE()
# coefficient_outline = agnes_.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("NYSK 数据集的 AGNES 算法：")
# begin_time = time.time()
# agnes_ = agnes.AGNES(
#     data_path=root_path_NYSK + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     cal_distance=dtools.cos_distance)
# agnes_.cluster()
# end_time = time.time()
# SSE_value = agnes_.get_SSE()
# coefficient_outline = agnes_.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

########## FCM ###########################

# for k in range(11)[10:]:
#     for m in range(4)[2:]:
#         f_cm = fcm.FCM(
#             data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#             # data_path=root_path_20_newsgroup + "weights_after_pca.csv",
#             cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
#             # cross_distance_path=root_path_20_newsgroup + "cross_euclidean_distance_feature_vector.csv",
#             # cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_weights.csv",
#             # cross_distance_path=root_path_20_newsgroup + "cross_euclidean_distance_weights.csv",
#             K=k,
#             M=m,
#             cal_distance=dtools.cos_distance)
#             # cal_distance=dtools.euclidean_distance)
#         f_cm.cluster()
#         SSE_value = f_cm.get_SSE()
#         coefficient_outline = f_cm.get_coefficient_outline()
#         ftools.append_txt(root_path_result + "20_Newsgroups_fcm.txt",
#                           [str(k) + "," + str(m) + ",feature_vector,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#         # ftools.append_txt(root_path_result + "20_Newsgroups_fcm.txt",
#         #                   [str(k) + "," + str(m) + ",feature_vector,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#         # ftools.append_txt(root_path_result + "20_Newsgroups_fcm.txt",
#         #                   [str(k) + "," + str(m) + ",weights,cos_distance," + str(SSE_value) + "," + str(coefficient_outline)])
#         # ftools.append_txt(root_path_result + "20_Newsgroups_fcm.txt",
#         #                   [str(k) + "," + str(m) + ",weights,euclidean_distance," + str(SSE_value) + "," + str(coefficient_outline)])

# print("20 newsgroups 数据集的 FCM 算法：")
# begin_time = time.time()
# f_cm = fcm.FCM(
#     data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     M=2,
#     cal_distance=dtools.cos_distance)
# f_cm.cluster()
# end_time = time.time()
# f_cm.show_cluster_allocation()
# SSE_value = f_cm.get_SSE()
# coefficient_outline = f_cm.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("AAAI 数据集的 FCM 算法：")
# begin_time = time.time()
# f_cm = fcm.FCM(
#     data_path=root_path_AAAI + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
#     K=7,
#     M=2,
#     cal_distance=dtools.cos_distance)
# f_cm.cluster()
# end_time = time.time()
# f_cm.show_cluster_allocation()
# SSE_value = f_cm.get_SSE()
# coefficient_outline = f_cm.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")

# print("NYSK 数据集的 FCM 算法：")
# begin_time = time.time()
# f_cm = fcm.FCM(
#     data_path=root_path_NYSK + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
#     K=6,
#     M=2,
#     cal_distance=dtools.cos_distance)
# f_cm.cluster()
# end_time = time.time()
# f_cm.show_cluster_allocation()
# SSE_value = f_cm.get_SSE()
# coefficient_outline = f_cm.get_coefficient_outline()
# print("SSE", SSE_value)
# print("轮廓系数", coefficient_outline)
# print("用时：", end_time - begin_time, "s")


########## DBSCAN ###########################

# db_scan = dbscan.DBSCAN(
#     data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_20_newsgroup + "cross_euclidean_distance_feature_vector.csv",
#     e=0.5,
#     minpts=2,
#     cal_distance=dtools.euclidean_distance)
# db_scan.cluster()
# db_scan.show_cluster_allocation()

# db_scan = dbscan.DBSCAN(
#     data_path=root_path_AAAI + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
#     e=0.5,
#     minpts=2,
#     cal_distance=dtools.cos_distance)
# db_scan.cluster()
# db_scan.show_cluster_allocation()

# db_scan = dbscan.DBSCAN(
#     data_path=root_path_NYSK + "feature_vector_after_pca.csv",
#     cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
#     e=0.5,
#     minpts=2,
#     cal_distance=dtools.cos_distance)
# db_scan.cluster()
# db_scan.show_cluster_allocation()
