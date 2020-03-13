import tools.file_tools as ftools
import numpy as np


class K_MEANS():

    def __init__(self, data_path, cross_distance_path, K, cal_distance):
        '''
        :param data_path: 数据文件的路径（string）
        :param cross_distance_path: 样本间交叉距离文件的路径（string）
        :param K: 聚簇成 K 类（int）
        :param cal_distance: 计算距离的方法（function）
        '''
        self.data_path = data_path
        self.cross_distance_path = cross_distance_path
        self.K = K
        self.cal_distance = cal_distance
        self.read_data()
        self.init_centroids()
        # self.init_centroids_plus()
        self.init_cluster_allocation()
        self.init_cross_distance()

    def read_data(self):
        '''
        读取数据文件
        self.data_mat为 sample_num*feature_num 的矩阵
        '''
        self.data_mat = ftools.read_csv_mat(self.data_path)
        self.sample_num = self.data_mat.shape[0]
        self.feature_num = self.data_mat.shape[1]

    def init_centroids(self):
        '''
        初始化聚簇中心
        self.centroids为 K*feature_num 的矩阵
        '''
        max_feature = np.max(self.data_mat, axis=0)
        min_feature = np.min(self.data_mat, axis=0)
        random_feature = np.random.random([self.K, self.feature_num])
        self.centroids = np.mat(min_feature + np.multiply(random_feature, (max_feature - min_feature)))

    def init_centroids_plus(self):
        '''
        k-means++算法初始化聚簇中心
        self.centroids为 K*feature_num 的矩阵
        '''
        self.centroids = np.mat(np.zeros([self.K, self.feature_num]))
        random_index = np.random.randint(self.sample_num)
        self.centroids[0,] = self.data_mat[random_index,]  # 随机初始化第一个聚簇中心
        for i in range(self.K)[1:]:
            dist_to_centroid = [0.0 for _ in range(self.sample_num)]
            sum_all = 0
            for j in range(self.sample_num):
                min_dist_to_centroid = self.cal_distance(self.centroids[0,], self.data_mat[j,])
                for z in range(i):
                    temp_dist = self.cal_distance(self.centroids[0,], self.data_mat[j,])
                    if temp_dist < min_dist_to_centroid:
                        min_dist_to_centroid = temp_dist
                dist_to_centroid[j] = min_dist_to_centroid
                sum_all += min_dist_to_centroid
            sum_all *= np.random.rand()
            for j, di in enumerate(dist_to_centroid):
                sum_all = sum_all - di
                if sum_all > 0:
                    continue
                self.centroids[i,] = self.data_mat[j,]
                break

    def init_cluster_allocation(self):
        '''
        初始化聚簇分配情况
        self.cluster_allocation为 sample_num*3 的矩阵
        self.cluster_allocation[i,0]为第 i 个样本分到的簇的索引
        self.cluster_allocation[i,1]为第 i 个样本到簇中心的距离
        self.cluster_allocation[i,2]为第 i 个样本的索引
        '''
        self.cluster_allocation = np.mat(np.zeros([self.sample_num, 3]))
        self.cluster_allocation[:, 2] = np.mat([i for i in range(self.sample_num)]).T

    def init_cross_distance(self):
        '''
        初试化样本间的交叉距离
        '''
        self.cross_distance = ftools.read_csv_array(self.cross_distance_path)

    def cluster(self):
        '''
        聚簇迭代
        '''
        changed = True  # 用于判断是否收敛
        iterate_num = 0  # 迭代次数
        while changed:
            iterate_num += 1
            # print("---第" + str(iterate_num) + "次迭代------------------------------------")
            changed = False
            for i in range(self.sample_num):
                sample = self.data_mat[i]
                min_dist = self.cal_distance(sample, self.centroids[0])
                min_cent = 0
                for j in range(self.K):
                    cent = self.centroids[j]
                    dist = self.cal_distance(sample, cent)
                    if dist < min_dist:
                        min_dist = dist
                        min_cent = j
                if not self.cluster_allocation[i, 0] == min_cent:
                    changed = True
                    # print("+++++++第" + str(i) + "个样本聚到" + str(min_cent) + "中+++++++++")
                self.cluster_allocation[i, 0] = min_cent
                self.cluster_allocation[i, 1] = min_dist
            self.cal_centroids()
            # self.show_cluster_allocation()

    def cal_centroids(self):
        '''
        每次迭代后，重新计算聚簇中心
        '''
        for i in range(self.K):
            data_in_allocation = self.data_mat[np.nonzero(self.cluster_allocation[:, 0] == i)[0]]
            self.centroids[i, :] = np.mean(data_in_allocation, axis=0)

    def show_cluster_allocation(self):
        '''
        统计并输出分类详情
        '''
        self.cluster_allocation_detail = []
        for i in range(self.K):
            i_allocation = self.cluster_allocation[np.nonzero(self.cluster_allocation[:, 0] == i)[0]]
            self.cluster_allocation_detail.append([int(j) for j in i_allocation[:, 2].T.tolist()[0]])
            print("第" + str(i) + "个簇：", "共" + str(len(self.cluster_allocation_detail[i])) + "个样本",
                  self.cluster_allocation_detail[i])

    def get_coefficient_outline(self):
        '''
        计算轮廓系数
        '''
        coefficient_outline_list = []
        for one_cluster in self.cluster_allocation_detail:
            for one_sample in one_cluster:
                distance_sum_a = 0.
                distance_sum_b = 0.
                for other_sample in one_cluster:
                    if one_sample == other_sample:
                        continue
                    distance_sum_a += self.cross_distance[one_sample, other_sample]
                other_cluster = []
                for c in self.cluster_allocation_detail:
                    if c == one_cluster:
                        continue
                    other_cluster.extend(c)
                for other_sample in other_cluster:
                    distance_sum_b += self.cross_distance[one_sample, other_sample]
                if len(one_cluster) - 1 == 0:
                    distance_sum_a = 0.
                else:
                    distance_sum_a /= len(one_cluster) - 1
                if self.sample_num - len(one_cluster) == 0:
                    distance_sum_b = 0.
                else:
                    distance_sum_b /= self.sample_num - len(one_cluster)
                coefficient_outline_list.append((distance_sum_b - distance_sum_a) / max(distance_sum_b, distance_sum_a))
        coefficient_outline = np.mean(np.array(coefficient_outline_list))
        return coefficient_outline

    def get_SSE(self):
        '''
        计算误差平方和 SSE
        '''
        return np.sum(self.cluster_allocation[:, 1])
