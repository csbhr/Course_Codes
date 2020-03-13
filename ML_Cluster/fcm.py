import numpy as np


class FCM():

    def __init__(self, data_mat, K, dist_type="euc", M=2):
        '''
        args:
            data_mat: np.array, with shape=[sample_num, feature_num]
            K: int32, the number of class
            dist_type: the type of calculate distance
                if dist_type = "euc", use euclidean distance
                if dist_type = "cos", use cos distance
            M: int32, flexible coefficient
        '''
        self.data_mat = data_mat
        self.sample_num = self.data_mat.shape[0]
        self.feature_num = self.data_mat.shape[1]
        self.K = K
        self.M = M
        if dist_type == "euc":
            self.cal_distance = self.euclidean_distance
        elif dist_type == "cos":
            self.cal_distance = self.cos_distance
        else:
            raise Exception("Message: no support dist_type")
        self.membership_mat = self.init_membership_mat()
        self.centroids = self.cal_centroids()
        self.cluster_allocation = self.cal_cluster_allocation()
        self.assessed_value = self.cal_assessed_value()
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

    def init_membership_mat(self):
        '''
        init membership mat

        returns:
            numpy.array, with shape=[sample_num, K]
        '''
        membership_list = []
        for i in range(self.sample_num):
            random_num_list = [np.random.rand() for _ in range(self.K)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]  # 首先归一化
            membership_list.append(temp_list)
        return np.array(membership_list)

    def cal_centroids(self):
        '''
        calculate cluster centroids refer to membership mat

        returns:
            numpy.array, with shape=[K, feature_num]
        '''
        centroids_list = []
        for i in range(self.K):
            mem = self.membership_mat[:, i]
            arise = np.power(mem, self.M)
            sum_arise = np.sum(arise)
            arise_tile = np.tile(arise, [self.feature_num, 1]).T
            mole = np.sum(np.multiply(self.data_mat, arise_tile), axis=0)
            cent = mole / sum_arise
            centroids_list.append(cent.tolist()[0])
        return np.array(centroids_list)

    def cal_cluster_allocation(self):
        '''
        calculate cluster allocation

        returns:
            numpy.array, with shape=[sample_num, 2]
                cluster_allocation[i,0]: the index of cluster which the sample allocate to
                cluster_allocation[i,1]: the index of the sample
        '''
        cluster_allocation = np.mat(np.zeros([self.sample_num, 2]))
        cluster_allocation[:, 0] = np.reshape(np.argmax(self.membership_mat, axis=1), [self.sample_num, 1])
        cluster_allocation[:, 1] = np.reshape(np.array([i for i in range(self.sample_num)]).T, [self.sample_num, 1])
        return cluster_allocation

    def update_membership_mat(self):
        '''
        update membership mat
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
        calculate the objective function: assessed value

        returns:
            numpy.float64, the assessed value
        '''
        x_u_distance = np.zeros([self.sample_num, self.K])
        for i in range(self.sample_num):
            for j in range(self.K):
                x_u_distance[i, j] = np.power(self.cal_distance(self.data_mat[i, :], self.centroids[j]), 2)
        return np.sum(np.multiply(self.membership_mat, x_u_distance))

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
            cluster_allocation_detail.append([int(j) for j in i_allocation[:, 1].T.tolist()[0]])
        return cluster_allocation_detail

    def get_assessed_value(self):
        '''
        get the assessed value

        returns:
            numpy.float64, the assessed value
        '''
        return self.assessed_value

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
