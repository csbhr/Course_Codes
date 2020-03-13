import numpy as np


class K_MEANS():

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
        self.centroids = self.init_centroids()
        self.cluster_allocation = self.init_cluster_allocation()
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

    def init_centroids(self):
        '''
        init cluster centroids

        returns:
            numpy.array, with shape=[K, feature_num]
        '''
        max_feature = np.max(self.data_mat, axis=0)
        min_feature = np.min(self.data_mat, axis=0)
        random_feature = np.random.random([self.K, self.feature_num])
        return np.array(min_feature + np.multiply(random_feature, (max_feature - min_feature)))

    def init_cluster_allocation(self):
        '''
        init cluster allocation

        returns:
            numpy.array, with shape=[sample_num, 3]
                cluster_allocation[i,0]: the index of cluster which the sample allocate to
                cluster_allocation[i,1]: the distance of the sample with cluster centroid
                cluster_allocation[i,2]: the index of the sample
        '''
        cluster_allocation = np.mat(np.zeros([self.sample_num, 3]))
        cluster_allocation[:, 2] = np.array([[i for i in range(self.sample_num)]]).T
        return cluster_allocation

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

    def cal_centroids(self):
        '''
        recalculate cluster centroids every cluster iteration
        '''
        for i in range(self.K):
            data_in_allocation = self.data_mat[np.nonzero(self.cluster_allocation[:, 0] == i)[0]]
            self.centroids[i, :] = np.mean(data_in_allocation, axis=0)

    def cluster(self):
        '''
        cluster iteration
        '''
        changed = True  # to jude whether convergent
        iterate_num = 0
        while changed:
            iterate_num += 1
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
                self.cluster_allocation[i, 0] = min_cent
                self.cluster_allocation[i, 1] = min_dist
            self.cal_centroids()

    def get_cluster_allocation(self):
        '''
        calculate cluster allocation detail

        returns:
            list, cluster allocation detail
                the len of list is the number of clusters
                the item of list is the samples' index in this cluster
        '''
        cluster_allocation_detail = []
        for i in range(self.K):
            i_allocation = self.cluster_allocation[np.nonzero(self.cluster_allocation[:, 0] == i)[0]]
            cluster_allocation_detail.append([int(j) for j in i_allocation[:, 2].T.tolist()[0]])
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
        return np.sum(self.cluster_allocation[:, 1])
