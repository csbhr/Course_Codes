import tools.file_tools as ftools
import matplotlib.pyplot as plt
import numpy as np


def pic_lines_1(x_value, y1_value, y2_value):
    width = 0.8
    plt.bar(x_value, y1_value, width=width, label="Silhouette Coefficient")
    plt.bar(x_value + width, y2_value, width=width, label="Time")
    plt.legend()
    plt.show()


def pic_lines_2(x_value, y1_value, y2_value):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(x_value, y1_value, color='red', marker='o', label="Silhouette Coefficient")
    ax1.set_ylabel("Silhouette Coefficient")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x_value, y2_value, color='blue', marker='x', label="TIme")
    ax2.set_ylabel("Time")
    # ax2.legend()

    plt.show()


root_path = "D:/WorkSpace/python/project_data/Data_Mining/result/"

##### K-Means ##############################

# data = ftools.read_txt(root_path + "20_Newsgroups_kmeans.txt")
# data = ftools.read_txt(root_path + "AAAI_kmeans.txt")
# data = ftools.read_txt(root_path + "NYSK_kmeans.txt")
#
# x_value = []
# y1_value = []
# y2_value = []
#
# for da in data[27:]:
#     da_list = da.split(",")
#     x_value.append(int(da_list[0]))
#     y1_value.append((float(da_list[3])))
#     y2_value.append((float(da_list[4])))
#
# pic_lines_2(x_value, y1_value, y2_value)


##### 不同算法比较 ##############################


# 20 newsgroups
# x_value = ["K-means", "FCM", "AGNES"]
# y1_value = [0.042, 0.039, 0.010]
# y2_value = [5.9, 39.3, 502.2]

# AAAI
# x_value = ["K-means", "FCM", "AGNES"]
# y1_value = [0.049, 0.052, 0.023]
# y2_value = [0.7, 7.2, 4.2]

# NYSK
x_value = ["K-means", "FCM", "AGNES"]
y1_value = [0.080, 0.072, 0.013]
y2_value = [55.0, 185.9, 71838.5]

pic_lines_2(x_value, y1_value, y2_value)
