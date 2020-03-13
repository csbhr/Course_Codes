import tools.file_tools as ftools
import numpy as np


class DBSCAN():

    def __init__(self, data_path, cross_distance_path, e, minpts, cal_distance):
        '''
        :param data_path: 数据文件的路径（string）
        :param cross_distance_path: 样本间交叉距离文件的路径（string）
        :param e: ε-邻域的距离阈值（float）
        :param minpts: ε-邻域的样本个数阈值（int）
        :param cal_distance: 计算距离的方法（function）
        '''
        self.data_path = data_path
        self.cross_distance_path = cross_distance_path
        self.e = e
        self.minpts = minpts
        self.cal_distance = cal_distance
        self.read_data()
        self.init_centroids()

    def read_data(self):
        '''
        读取数据文件
        self.data_list为 sample_num*feature_num 的list
        '''
        data_temp = ftools.read_csv_list(self.data_path)
        self.data_list = []
        for i in data_temp:
            self.data_list.append(tuple(i))
        self.sample_num = len(self.data_list)
        self.feature_num = len(self.data_list[0])

    def init_centroids(self):
        '''
        初始化：
            核心对象集合 T
            聚类个数 K
            聚类集合 C
            未访问集合 P
        '''
        self.T = set()
        self.K = 0
        self.C = []
        self.P = set(self.data_list)
        for sample in self.data_list:
            e_neighbor = [i for i in self.data_list if self.cal_distance(np.mat(sample), np.mat(i)) <= self.e]
            if len(e_neighbor) >= self.minpts:
                self.T.add(sample)

    def init_cross_distance(self):
        '''
        初试化样本间的交叉距离
        '''
        self.cross_distance = ftools.read_csv_array(self.cross_distance_path)

    def cluster(self):
        '''
        聚簇迭代
        '''
        while len(self.T):
            P_old = self.P.copy()
            o = list(self.T)[np.random.randint(0, len(self.T))]
            self.P = self.P - set(o)
            Q = []
            Q.append(o)
            while len(Q):
                q = Q[0]
                e_neighbor_q = [i for i in self.data_list if self.cal_distance(np.mat(q), np.mat(i)) <= self.e]
                if len(e_neighbor_q) >= self.minpts:
                    S = self.P & set(e_neighbor_q)
                    Q += (list(S))
                    self.P = self.P - S
                Q.remove(q)
            self.K += 1
            C_k = list(P_old - self.P)
            self.T = self.T - set(C_k)
            self.C.append(C_k)

    def show_cluster_allocation(self):
        '''
        统计并输出分类详情
        '''
        for i in range(len(self.C)):
            print("第" + str(i) + "个簇：", "共" + str(len(self.C[i])) + "个样本", self.C[i])
