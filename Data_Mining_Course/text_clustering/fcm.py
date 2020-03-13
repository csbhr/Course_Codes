import tools.file_tools as ftools
import numpy as np


class FCM():

    def __init__(self, data_path, cross_distance_path, K, M, cal_distance):
        '''
        :param data_path: 数据文件的路径（string）
        :param cross_distance_path: 样本间交叉距离文件的路径（string）
        :param K: 聚簇成 K 类（int）
        :param M: 柔性参数（int）
        :param cal_distance: 计算距离的方法（function）
        '''
        self.data_path = data_path
        self.cross_distance_path = cross_distance_path
        self.K = K
        self.M = M
        self.cal_distance = cal_distance
        self.read_data()
        self.init_membership_mat()
        self.cal_centroids()
        self.cal_cluster_allocation()
        self.cal_assessed_value()
        self.init_cross_distance()

    def read_data(self):
        '''
        读取数据文件
        self.data_mat为 sample_num*feature_num 的矩阵
        '''
        self.data_mat = ftools.read_csv_mat(self.data_path)
        self.sample_num = self.data_mat.shape[0]
        self.feature_num = self.data_mat.shape[1]

    def init_membership_mat(self):
        '''
        初始化模糊矩阵
        self.membership_mat为 sample_num*K 的矩阵
        '''
        membership_list = []
        for i in range(self.sample_num):
            random_num_list = [np.random.rand() for _ in range(self.K)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]  # 首先归一化
            membership_list.append(temp_list)
        self.membership_mat = np.mat(membership_list)

    def cal_centroids(self):
        '''
        根据模糊矩阵计算聚簇中心
        self.centroids为 K*feature_num 的矩阵
        '''
        centroids_list = []
        for i in range(self.K):
            mem = self.membership_mat[:, i]
            arise = np.power(mem, self.M)
            sum_arise = np.sum(arise)
            mole = np.sum(np.multiply(self.data_mat, arise), axis=0)
            cent = mole / sum_arise
            centroids_list.append(cent.tolist()[0])
        self.centroids = np.mat(centroids_list)

    def cal_cluster_allocation(self):
        '''
        初始化聚簇分配情况
        self.cluster_allocation为 sample_num*2 的矩阵
        self.cluster_allocation[i,0]为第 i 个样本分到的簇的索引
        self.cluster_allocation[i,1]为第 i 个样本的索引
        '''
        self.cluster_allocation = np.mat(np.zeros([self.sample_num, 2]))
        self.cluster_allocation[:, 0] = np.argmax(self.membership_mat, axis=1)
        self.cluster_allocation[:, 1] = np.mat([i for i in range(self.sample_num)]).T

    def update_membership_mat(self):
        '''
        更新模糊矩阵
        '''
        for i in range(self.sample_num):
            x_i = self.data_mat[i, :]
            centroids_list = self.centroids.tolist()
            x_u_distance = [self.cal_distance(x_i, np.mat(cent)) for cent in centroids_list]
            for j in range(self.K):
                den = sum([np.power(x_u_distance[j] / x_u_distance[c], 2 / (self.M - 1)) for c in range(self.K)])
                self.membership_mat[i, j] = 1 / den

    def cal_assessed_value(self):
        '''
        计算目标函数
        self.assessed_value是 float 类型
        '''
        x_u_distance = np.zeros([self.sample_num, self.K])
        for i in range(self.sample_num):
            for j in range(self.K):
                x_u_distance[i, j] = np.power(self.cal_distance(self.data_mat[i, :], self.centroids[j, :]), 2)
        self.assessed_value = np.sum(np.multiply(self.membership_mat, x_u_distance))

    def init_cross_distance(self):
        '''
        初试化样本间的交叉距离
        '''
        self.cross_distance = ftools.read_csv_array(self.cross_distance_path)

    def cluster(self):
        '''
        聚簇迭代
        '''
        iterate_num = 0
        diff = 9999
        before_assessed_value = self.assessed_value
        # while iterate_num<100:
        while diff > 1e-3 and iterate_num < 50:
            self.update_membership_mat()
            self.cal_centroids()
            self.cal_cluster_allocation()
            self.cal_assessed_value()
            diff = np.abs(before_assessed_value - self.assessed_value)
            before_assessed_value = self.assessed_value
            iterate_num += 1
            # print("---第" + str(iterate_num) + "次迭代------------------------------------")
            # self.show_cluster_allocation()

    def show_cluster_allocation(self):
        '''
        统计并输出分类详情
        '''
        self.cluster_allocation_detail = []
        for i in range(self.K):
            i_allocation = self.cluster_allocation[np.nonzero(self.cluster_allocation[:, 0] == i)[0]]
            self.cluster_allocation_detail.append([int(j) for j in i_allocation[:, 1].T.tolist()[0]])
            print("第" + str(i) + "个簇：", "共" + str(len(self.cluster_allocation_detail[i])) + "个样本",
                  self.cluster_allocation_detail[i])
        print("目标函数：", self.assessed_value)

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
        SSE_value = 0.
        for one_cluster in self.cluster_allocation_detail:
            this_cluster_samples = self.data_mat[one_cluster]
            this_cluster_centroids = np.mean(this_cluster_samples, axis=0)
            for one_index in one_cluster:
                SSE_value += self.cal_distance(self.data_mat[one_index], this_cluster_centroids)
        return SSE_value
