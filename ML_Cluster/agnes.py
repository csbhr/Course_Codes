import numpy as np


class Cluster_Node():

    def __init__(self, centre, index, left_node=None, right_node=None):
        '''
        cluster node

        args:
            centre: np.array, with shape=[feature_num],
                the centre sample which can represent the node
            index: int32, the index of the node
            left_node: Cluster_Node, the left child of the node
            right_node: Cluster_Node, the right child of the node
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


class AGNES():

    def __init__(self, data_mat, K, dist_type="euc"):
        '''
        args:
            data_mat: np.array, with shape=[sample_num, feature_num]
            K: int32, the number of class
            dist_type: the type of calculate distance
                if dist_type = "euc", use euclidean distance
                if dist_type = "cos", use cos distance
        '''
        self.data_mat = data_mat
        self.sample_num = self.data_mat.shape[0]
        self.feature_num = self.data_mat.shape[1]
        self.K = K
        if dist_type == "euc":
            self.cal_distance = self.euclidean_distance
        elif dist_type == "cos":
            self.cal_distance = self.cos_distance
        else:
            raise Exception("Message: no support dist_type")
        self.node_index_now, self.cluster_nodes = self.init_cluster_nodes()
        self.cross_distance = self.init_cross_distance()

    def euclidean_distance(self, vector_1, vector_2):
        '''
        calculate euclidean distance

        args:
            vector_1: numpy.array, with shape = [n]
            vector_2: numpy.array, with shape = [n]

        returns:
            numpy.float64, euclidean distance
        '''
        sub = vector_1 - vector_2
        dist = np.sqrt(np.dot(sub, sub.T))
        return dist

    def cos_distance(self, vector_1, vector_2):
        '''
        calculate cos distance

        args:
            vector_1: numpy.array, with shape = [n]
            vector_2: numpy.array, with shape = [n]

        returns:
            numpy.float64, 1-cos(vector_1, vector_2)
        '''
        sum_1 = float(np.dot(vector_1, vector_2.T))
        sum_2 = float(np.sqrt(np.dot(vector_1, vector_1.T)))
        sum_3 = float(np.sqrt(np.dot(vector_2, vector_2.T)))
        return 1 - sum_1 / (sum_2 * sum_3)

    def init_cluster_nodes(self):
        '''
        init cluster nodes

        returns:
            node_index_now: int32
            cluster_nodes: list, the item of list are the cluster nodes
        '''
        node_index_now = 0
        cluster_nodes = []
        for i in range(self.sample_num):
            node = Cluster_Node(centre=self.data_mat[i], index=i)
            cluster_nodes.append((node_index_now, node))
            node_index_now += 1
        return node_index_now, cluster_nodes

    def init_cross_distance(self):
        '''
        init the cross distance between samples

        returns:
            numpy.array, with shape=[sample_num, sample_num]
        '''
        sample_num = self.data_mat.shape[0]
        cross_distance = np.zeros([sample_num, sample_num])
        for i in range(sample_num):
            for j in range(sample_num):
                cross_distance[i, j] = self.cal_distance(self.data_mat[i, :], self.data_mat[j, :])
        return cross_distance

    def cluster(self):
        '''
        cluster iteration
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

    def get_cluster_allocation(self):
        '''
        calculate cluster allocation detail

        returns:
            list, cluster allocation detail
                the len of list is the number of clusters
                the item of list is the samples' index in this cluster
        '''
        cluster_allocation_detail = []
        for node_tuple in self.cluster_nodes:
            node = node_tuple[1]
            one_cluster = []
            for leaf in node.leaf_nodes:
                one_cluster.append(leaf[0])
            cluster_allocation_detail.append(one_cluster)
        return cluster_allocation_detail

    def get_coefficient_outline(self):
        '''
        calculate coefficient outline

        returns:
            numpy.float64, the coefficient outline
        '''
        coefficient_outline_list = []
        cluster_allocation_detail = self.get_cluster_allocation()
        for one_cluster in cluster_allocation_detail:
            for one_sample in one_cluster:
                distance_sum_a = 0.
                distance_sum_b = 0.
                for other_sample in one_cluster:
                    if one_sample == other_sample:
                        continue
                    distance_sum_a += self.cross_distance[one_sample, other_sample]
                other_cluster = []
                for c in cluster_allocation_detail:
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
        calculate sum of squared errors(SSE)

        returns:
            numpy.float64, the SSE
        '''
        SSE_value = 0.
        cluster_allocation_detail = self.get_cluster_allocation()
        for one_cluster in cluster_allocation_detail:
            this_cluster_samples = self.data_mat[one_cluster]
            this_cluster_centroids = np.mean(this_cluster_samples, axis=0)
            for one_index in one_cluster:
                SSE_value += self.cal_distance(self.data_mat[one_index], this_cluster_centroids)
        return SSE_value
