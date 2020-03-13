import text_clustering.k_means as k_means
import text_clustering.fcm as fcm
import text_clustering.agnes as agnes
import tools.distance_tools as dtools
import time

root_path_20_newsgroup = "data/20 Newsgroups/"
root_path_AAAI = "data/AAAI-14 Accepted Papers - Papers/"
root_path_NYSK = "data/NYSK Data Set/"

while True:
    print("----------------------------------------------------")
    print("-                    MENU                          -")
    print("-       1. 20 Newsgroups using K-means             -")
    print("-       2. 20 Newsgroups using AGNES               -")
    print("-       3. 20 Newsgroups using FCM                 -")
    print("-       4. AAAI-14 using K-means                   -")
    print("-       5. AAAI-14 using AGNES                     -")
    print("-       6. AAAI-14 using FCM                       -")
    print("-       7. NYSK using K-means                      -")
    print("-       8. NYSK using AGNES                        -")
    print("-       9. NYSK using FCM                          -")
    print("-       0. EXIT                                    -")
    print("----------------------------------------------------")
    print("Please input the command:")
    command = input()
    if command == "1":
        print("20 newsgroups 数据集的 K-means 算法：")
        begin_time = time.time()
        kmeans = k_means.K_MEANS(
            data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
            K=6,
            cal_distance=dtools.cos_distance)
        kmeans.cluster()
        end_time = time.time()
        kmeans.show_cluster_allocation()
        SSE_value = kmeans.get_SSE()
        coefficient_outline = kmeans.get_coefficient_outline()
        print("SSE：", SSE_value)
        print("轮廓系数：", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "2":
        print("20 newsgroups 数据集的 AGNES 算法：")
        begin_time = time.time()
        agnes_ = agnes.AGNES(
            data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
            K=6,
            cal_distance=dtools.cos_distance)
        agnes_.cluster()
        end_time = time.time()
        SSE_value = agnes_.get_SSE()
        coefficient_outline = agnes_.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "3":
        print("20 newsgroups 数据集的 FCM 算法：")
        begin_time = time.time()
        f_cm = fcm.FCM(
            data_path=root_path_20_newsgroup + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_20_newsgroup + "cross_cos_distance_feature_vector.csv",
            K=6,
            M=2,
            cal_distance=dtools.cos_distance)
        f_cm.cluster()
        end_time = time.time()
        f_cm.show_cluster_allocation()
        SSE_value = f_cm.get_SSE()
        coefficient_outline = f_cm.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "4":
        print("AAAI-14 数据集的 K-means 算法：")
        begin_time = time.time()
        kmeans = k_means.K_MEANS(
            data_path=root_path_AAAI + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
            K=7,
            cal_distance=dtools.cos_distance)
        kmeans.cluster()
        end_time = time.time()
        kmeans.show_cluster_allocation()
        SSE_value = kmeans.get_SSE()
        coefficient_outline = kmeans.get_coefficient_outline()
        print("SSE：", SSE_value)
        print("轮廓系数：", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "5":
        print("AAAI 数据集的 AGNES 算法：")
        begin_time = time.time()
        agnes_ = agnes.AGNES(
            data_path=root_path_AAAI + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
            K=7,
            cal_distance=dtools.cos_distance)
        agnes_.cluster()
        end_time = time.time()
        SSE_value = agnes_.get_SSE()
        coefficient_outline = agnes_.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "6":
        print("AAAI 数据集的 FCM 算法：")
        begin_time = time.time()
        f_cm = fcm.FCM(
            data_path=root_path_AAAI + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_AAAI + "cross_cos_distance_feature_vector.csv",
            K=7,
            M=2,
            cal_distance=dtools.cos_distance)
        f_cm.cluster()
        end_time = time.time()
        f_cm.show_cluster_allocation()
        SSE_value = f_cm.get_SSE()
        coefficient_outline = f_cm.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "7":
        print("NYSK 数据集的 K-means 算法：")
        begin_time = time.time()
        kmeans = k_means.K_MEANS(
            data_path=root_path_NYSK + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
            K=6,
            cal_distance=dtools.cos_distance)
        kmeans.cluster()
        end_time = time.time()
        kmeans.show_cluster_allocation()
        SSE_value = kmeans.get_SSE()
        coefficient_outline = kmeans.get_coefficient_outline()
        print("SSE：", SSE_value)
        print("轮廓系数：", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "8":
        print("NYSK 数据集的 AGNES 算法：")
        begin_time = time.time()
        agnes_ = agnes.AGNES(
            data_path=root_path_NYSK + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
            K=6,
            cal_distance=dtools.cos_distance)
        agnes_.cluster()
        end_time = time.time()
        SSE_value = agnes_.get_SSE()
        coefficient_outline = agnes_.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "9":
        print("NYSK 数据集的 FCM 算法：")
        begin_time = time.time()
        f_cm = fcm.FCM(
            data_path=root_path_NYSK + "feature_vector_after_pca.csv",
            cross_distance_path=root_path_NYSK + "cross_cos_distance_feature_vector.csv",
            K=6,
            M=2,
            cal_distance=dtools.cos_distance)
        f_cm.cluster()
        end_time = time.time()
        f_cm.show_cluster_allocation()
        SSE_value = f_cm.get_SSE()
        coefficient_outline = f_cm.get_coefficient_outline()
        print("SSE", SSE_value)
        print("轮廓系数", coefficient_outline)
        print("用时：", end_time - begin_time, "s")
    elif command == "0":
        break
    else:
        print("Please input right command!")
