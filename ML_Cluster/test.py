import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from k_means import K_MEANS
from fcm import FCM
from agnes import AGNES

iris = load_iris()
iris_data = np.array(iris.data)


def figure_result(data, alloc):
    x_axis, y_axis = 0, 1
    class_1 = {
        "x": [],
        "y": []
    }
    class_2 = {
        "x": [],
        "y": []
    }
    class_3 = {
        "x": [],
        "y": []
    }
    for i in alloc[0]:
        class_1["x"].append(data[i, x_axis])
        class_1["y"].append(data[i, y_axis])
    for i in alloc[1]:
        class_2["x"].append(data[i, x_axis])
        class_2["y"].append(data[i, y_axis])
    for i in alloc[2]:
        class_3["x"].append(data[i, x_axis])
        class_3["y"].append(data[i, y_axis])
    plt.scatter(class_1["x"], class_1["y"], c="red")
    plt.scatter(class_2["x"], class_2["y"], c="yellow")
    plt.scatter(class_3["x"], class_3["y"], c="blue")
    plt.show()


def figure_elbow(SSE_elbow_method):
    num = len(SSE_elbow_method)
    xs = [i + 1 for i in range(num)]
    ys = SSE_elbow_method
    plt.plot(xs, ys, color='red', marker="x", label="SSE")
    plt.legend()
    plt.show()

# # K_Means算法
# start_time = time.time()
# k_m = K_MEANS(data_mat=iris_data, K=3)
# k_m.cluster()
# end_time = time.time()
# alloc = k_m.get_cluster_allocation()
# for a in alloc:
#     print(a)
# print("轮廓系数：", k_m.get_coefficient_outline())
# print("误差平方和SSE：", k_m.get_SSE())
# print("算法时间：", end_time - start_time, "s")
# figure_result(iris_data, alloc)

# AGNES算法
start_time = time.time()
ag = AGNES(data_mat=iris_data, K=3)
ag.cluster()
end_time = time.time()
alloc = ag.get_cluster_allocation()
for a in alloc:
    print(a)
print("轮廓系数：", ag.get_coefficient_outline())
print("误差平方和SSE：", ag.get_SSE())
print("算法时间：", end_time - start_time, "s")
figure_result(iris_data, alloc)

# # 画手肘法的图
# SSE_elbow_method = [291.61, 263.75, 109.78, 106.68, 99.30, 98.01, 96.66, 95.93]
# figure_elbow(SSE_elbow_method)
