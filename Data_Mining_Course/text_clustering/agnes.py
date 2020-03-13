import tools.file_tools as ftools
import numpy as np


class Cluster_Node():

    def __init__(self, centre, index, left_node=None, right_node=None):
        '''
        聚簇结点
        :param centre: 能代表此结点的中心（1*feature_num mat）
        :param index: 如果是叶子结点，需要给出样本索引（int）
        :param left_node: 左孩子结点（Cluster_Node）
        :param right_node: 右孩子结点（Cluster_Node）
        '''
        self.centre = centre
        self.left_node = left_node
        self.right_node = right_node
        self.index = index

        self.leaf_nodes = []
        if self.index == -1:
            self.leaf_nodes.extend(self.left_node.leaf_nodes)
            self.leaf_nodes.extend(self.right_node.leaf_nodes)
        else:
            self.leaf_nodes = [[self.index, self.centre]]
        # if self.left_node == None and self.right_node == None:
        #     self.leaf_nodes = [[self.index, self.centre]]
        # else:
        #     if not self.left_node == None:
        #         self.leaf_nodes.extend(self.left_node.leaf_nodes)
        #     if not self.right_node == None:
        #         self.leaf_nodes.extend(self.right_node.leaf_nodes)


class AGNES():

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
        self.init_cluster_nodes()
        self.init_cross_distance()

    def read_data(self):
        '''
        读取数据文件
        self.data_mat为 sample_num*feature_num 的矩阵
        '''
        self.data_mat = ftools.read_csv_mat(self.data_path)
        self.sample_num = self.data_mat.shape[0]
        self.feature_num = self.data_mat.shape[1]

    def init_cluster_nodes(self):
        self.node_index_now = 0
        self.cluster_nodes = []
        for i in range(self.sample_num):
            node = Cluster_Node(centre=self.data_mat[i], index=i)
            self.cluster_nodes.append((self.node_index_now, node))
            self.node_index_now += 1

    def init_cross_distance(self):
        '''
        初试化样本间的交叉距离
        '''
        self.cross_distance = ftools.read_csv_array(self.cross_distance_path)

    def cluster(self):
        '''
        聚簇迭代
        '''
        distance_dict = {}
        while len(self.cluster_nodes) > self.K:
            min_distance_couple = (self.cluster_nodes[0], self.cluster_nodes[1])
            min_distance = self.cal_distance(self.cluster_nodes[0][1].centre, self.cluster_nodes[1][1].centre)
            for i in range(len(self.cluster_nodes) - 1):
                for j in range(i + 1, len(self.cluster_nodes)):
                    temp_couple_index = (self.cluster_nodes[i][0], self.cluster_nodes[j][0])
                    if temp_couple_index in distance_dict:
                        dis = distance_dict[temp_couple_index]
                    else:
                        dis = self.cal_distance(self.cluster_nodes[i][1].centre, self.cluster_nodes[j][1].centre)
                        distance_dict[temp_couple_index] = dis
                    if dis < min_distance:
                        min_distance = dis
                        min_distance_couple = (self.cluster_nodes[i], self.cluster_nodes[j])
            temp = np.vstack((min_distance_couple[0][1].centre, min_distance_couple[1][1].centre))
            new_node = Cluster_Node(centre=np.mean(temp, axis=0),
                                    index=-1,
                                    left_node=min_distance_couple[0][1],
                                    right_node=min_distance_couple[1][1])
            self.cluster_nodes.remove(min_distance_couple[0])
            self.cluster_nodes.remove(min_distance_couple[1])
            self.cluster_nodes.append((self.node_index_now, new_node))
            self.node_index_now += 1
        self.show_cluster_allocation()

    def show_cluster_allocation(self):
        '''
        统计并输出分类详情
        '''
        self.cluster_allocation_detail = []
        for node_tuple in self.cluster_nodes:
            node = node_tuple[1]
            one_cluster = []
            for leaf in node.leaf_nodes:
                one_cluster.append(leaf[0])
            self.cluster_allocation_detail.append(one_cluster)
        for i in range(len(self.cluster_allocation_detail)):
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
        SSE_value = 0.
        for one_cluster in self.cluster_allocation_detail:
            this_cluster_samples = self.data_mat[one_cluster]
            this_cluster_centroids = np.mean(this_cluster_samples, axis=0)
            for one_index in one_cluster:
                SSE_value += self.cal_distance(self.data_mat[one_index], this_cluster_centroids)
        return SSE_value
