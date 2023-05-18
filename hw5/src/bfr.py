import argparse
import pyspark
import time
import os
import sys
import math
import json


# represents of a type of set for which CS and DS are built off of in the BFR algorithm
# CS - compression set
# DS - discard set
class BfrSet:
    def __init__(self, location_data, data_indices):
        self.num_pts = len(location_data)

        # initialize list of totals and list of squared totals
        # squared_total is meant primarily for calculation of variance
        self.total = [0 for item in range(len(location_data[0]))]
        self.squared_total = self.total.copy()

        # an attempt to add actual values to totals and squared totals
        for data_pt in location_data:
            self.total = [w1 + w2 for w1, w2 in zip(self.total, data_pt)]
            self.squared_total = [w1 + w2 for w1, w2 in zip(self.squared_total, list(map(lambda item: item * item, data_pt)))]

        self.data_indices = [] + data_indices

    # returns coordinates of a centroid
    def find_centroid(self):
        return [i / self.num_pts for i in self.total]

    # variance of a set
    def compute_var(self):
        return [(square_n / self.num_pts) - (sum_n / self.num_pts) ** 2 for square_n, sum_n in
                zip(self.squared_total, self.total)]

    # add data point
    def add_data(self, data_pt, data_index):
        # update set with data
        self.num_pts += 1
        self.total = [w1 + w2 for w1, w2 in zip(self.total, data_pt)]
        self.squared_total = [w1 + w2 for w1, w2 in zip(self.squared_total, list(map(lambda x: x * x, data_pt)))]

        self.data_indices.append(data_index)

    # merge sets together!
    def add_bfr(self, other):
        self.num_pts += other.num_pts
        for i in range(len(self.total)):
            self.total[i] += other.total[i]
            self.squared_total[i] += other.squared_total[i]
        self.data_indices += other.data_indices

    # categorizes set based on centroid and variance
    # this is solely a quality-of-life improvement for programming this assignment!
    def __repr__(self):
        return f"centroid: {str(self.find_centroid())}, variance: {str(self.compute_var())}"

# represents RS set in BFR algorithm
# RS - retained set
class RS:
    def __init__(self, location_data, data_indices):
        self.location = {index: location_data[index]for index in data_indices}

    # adds location data for retained set
    def set_location(self, adding_location_data):
        for index, data in adding_location_data.items():
            self.location[index] = data


# represents an instance of the K-means algorithm for which the BFR algorithm builds off of
# assign points to "closest" (smallest mahalanobis distance) clusters
# iterates until convergence of cluster assignment OR total error
class KMeans:
    def __init__(self, data_pts, num_cluster):
        # data points and cluster indices
        self.k = num_cluster
        self.data_pts = data_pts

        # empty dataset has empty clusters and centroids
        if len(data_pts) == 0:
            self.clusters = []
            self.centroids = []
            return

        # keep track of data
        self.list = []
        self.indices = dict()

        # initialize based on data and its indices
        for i, (data_pt, v) in enumerate(self.data_pts.items()):
            self.list.append(v)
            self.indices[i] = data_pt

        self.num_pts = len(self.list)
        self.dims = len(self.list[0])

        # debug note: datasets with fewer clusters than data don't work properly
        # instead, only use as many clusters as possible
        if num_cluster >= len(data_pts):
            self.clusters = [[i] for i in range(self.num_pts)]
            self.centroids = self.list
            return

        # list of list of data index in X for each cluster
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []  # centroid vector of each cluster

        self.predict()

    # filter data based on cluster size
    def filter_data(self, thres):
        over, under = [], []
        for cluster in self.clusters:

            if len(cluster) > thres:
                over += list(map(lambda pt: self.indices[pt], cluster))
            else:
                under += list(map(lambda pt: self.indices[pt], cluster))

        return over, under

    # secondary step to split in which lone datapoints are filtered out
    def filter_split(self):
        over, under = [], []
        for cluster in self.get_cluster():

            if len(cluster) > 1:
                over.append(cluster)
            else:
                under.append(cluster)

        return over, under

    # creates cluster based on datapoints
    def get_cluster(self):
        return list(map(lambda pt: list(map(lambda coord: self.indices[coord], pt)), self.clusters))

    # returns data from a given cluster for use in next steps of kmeans
    def get_data_cluster(self):
        cluster_data = dict()
        for index, cluster in enumerate(self.clusters):

            for pt in cluster:
                data = self.indices[pt]
                cluster_data[data] = index

        return cluster_data

    # creates centroids from datapoints
    def create_centroids(self):
        self.centroids.append(self.list[0])
        while len(self.centroids) < self.k:

            dist_list = [min([compute_dist(pt, centroid) for centroid in self.centroids]) for pt in self.list]
            self.centroids.append(self.list[dist_list.index(max(dist_list))])


    def predict(self):
        self.create_centroids()
        # iterate high number of times
        for num in range(1000):
            # update clusters
            clusters = [[] for i in range(self.k)]
            for index, pt in enumerate(self.list):

                dists = [compute_dist(pt, centroid) for centroid in self.centroids]
                closest = dists.index(min(dists))

                clusters[closest].append(index)

            self.clusters = clusters
            original_centroids = self.centroids

            # update centroids
            centroids = []
            for cluster in self.clusters:
                v = [0 for i in range(self.dims)]

                for item in cluster:
                    v = [w1+w2 for w1,w2 in zip(v, self.list[item])]
                centroids.append([i / len(cluster) for i in v])

            self.centroids = centroids

            # stopping criteria
            dists = [compute_dist(original_centroids[i], self.centroids[i])for i in range(self.k)]
            if sum(dists) == 0:
                break


# returns euclidean distance between points
def compute_dist(data1, data2):
    total = 0
    for i in range(len(data1)):
        total += (data1[i] - data2[i]) ** 2

    return math.sqrt(total)

# returns mahalanobis distance between a point and some CS or DS
def compute_mahalanobis(data_pt, set_DS_CS):
    centroid = set_DS_CS.find_centroid()
    var = set_DS_CS.compute_var()
    vector = [((x_i - c_i) / math.sqrt(sigma_i)) ** 2 for x_i, c_i, sigma_i in zip(data_pt, centroid, var)]

    return math.sqrt(sum(vector))

# creates BfrSets for use as CS, DS
def create_set(location_data, cluster_data):
    return [BfrSet(list(map(lambda x: location_data[x], cluster)), cluster) for cluster in cluster_data]

# assign pts to either CS or DS
def assign_set(location_data, DS_CS):
    if len(DS_CS) == 0:
        return location_data

    assigned_data_indices = set()

    # get mahalanobis distance for every point
    for idx, data_pt in location_data.items():
        dists = [compute_mahalanobis(data_pt, set_DS_CS) for set_DS_CS in DS_CS]
        min_dist = min(dists)

        # min(distances) < 2sqrt(d) as stopping criteria
        if min_dist > math.sqrt(len(data_pt)) * 2:
            continue

        # update set
        DS_CS[dists.index(min_dist)].add_data(data_pt, idx)
        assigned_data_indices.add(idx)

    return {k: location_data[k] for k in set(location_data) - assigned_data_indices}

# combine CS with CS
def CS_combine_self(CS):
    while len(CS) > 1:
        merge_list = []

        for i in range(len(CS)):
            for j in range(i + 1, len(CS)):
                dist = max(compute_mahalanobis(CS[i].find_centroid(), CS[j]),compute_mahalanobis(CS[j].find_centroid(), CS[i]))
                merge_list.append((i, j, dist))
        merge_list.sort(key=lambda x: x[2])

        # debug note: terminate at correct time to avoid KeyError
        if merge_list[0][2] > math.sqrt(len(CS[0].total)) * 2:
            break

        # add item and pop from queue
        CS[merge_list[0][0]].add_bfr(CS[merge_list[0][1]])
        CS.pop(merge_list[0][1])

    return CS


# get num pts for CS or DS
def get_num_pts(DS_CS):
    return sum(map(lambda cluster: cluster.num_pts, DS_CS))

# combine CS with DS
def CS_combine_DS(DS, CS):
    for CS_item in CS:
        dists = [compute_dist(CS_item.find_centroid(), DS_item.find_centroid()) for DS_item in DS]
        merge_index = dists.index(min(dists))

        # add CS
        DS[merge_index].add_bfr(CS_item)

    return DS


def main(input_path, n_cluster, out_file2, out_file1, sc):
    # headings
    with open(out_file1, 'w') as outfile:
        print(
            "round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained",
            file=outfile)

    # iterate for every dataset file in directory
    for idx, data_path in enumerate(sorted(os.listdir(input_path))):

        # text_rdd - raw text input of location data with some preprocessing
        text_rdd = sc.textFile(input_path + "/" + data_path).map(lambda line: line.split(",")) \
            .map(lambda line: (line[0], list(map(lambda coordinate: float(coordinate), line[1:]))))
        location_data = text_rdd.collectAsMap()

        # first file used for random sampling
        if idx==0:
            sampling_data = text_rdd.sample(False, 30).collectAsMap()
            kmeans_sampling = KMeans(sampling_data, n_cluster)

            # filter data in K-means
            in_data, out_data = kmeans_sampling.filter_data(n_cluster)
            in_rdd = sc.parallelize(in_data).map(lambda point: (point, location_data[point]))
            k_in = KMeans(in_rdd.collectAsMap(), n_cluster)

            # use K-means data to create DS, CS, RS
            DS = create_set(location_data, k_in.get_cluster())
            if len(out_data) == 0:
                CS = []
                RS_list = RS(location_data, [])

            else:
                # out_rdd - points with location data after K-means
                out_rdd = sc.parallelize(out_data).map(lambda point: (point, location_data[point]))
                out = out_rdd.collectAsMap()

                # run K-means on result
                k_out = KMeans(out, n_cluster)

                # filter CS and RS clusters
                cs_clusters, rs_clusters = k_out.filter_split()
                CS = create_set(location_data, cs_clusters)

                # initialize collection of clusters and RS
                cluster_list = [cluster[0] for cluster in rs_clusters]
                RS_list = RS(location_data, cluster_list)

            location_data = {k: location_data[k] for k in set(location_data) - set(sampling_data)}

        # assign data to sets
        location_data = assign_set(location_data, DS)
        location_data = assign_set(location_data, CS)

        # initialize RS
        RS_list.set_location(location_data)
        k_RS = KMeans(RS_list.location, n_cluster)

        # split K-means data to create new CS and RS
        cs_clusters, rs_clusters = k_RS.filter_split()
        CS += create_set(RS_list.location, cs_clusters)

        cluster_list = [cluster[0] for cluster in rs_clusters]
        RS_list = RS(RS_list.location, cluster_list)
        CS = CS_combine_self(CS)

        # last file follows instruction 7m.
        # merge clusters in CS with DS clusters of Mahalanobis distance < 2sqrt(d)
        if idx == len(os.listdir(input_path)) - 1:
            DS = CS_combine_DS(DS, CS)
            RS_list.location = assign_set(RS_list.location, DS)
            CS = []


        # clusters output
        out1 = [idx + 1, len(DS), get_num_pts(DS),len(CS), get_num_pts(CS), len(RS_list.location)]
        with open(out_file1, 'a') as outfile:
            print(str(out1)[1: -1], file=outfile)

    # intermediate output
    out2 = {}
    for n, DS_n in enumerate(DS):
        for data_index in DS_n.data_indices:
            out2[data_index] = n

    for i in RS_list.location:
        out2[i] = -1

    with open(out_file2, 'w') as outfile:
        json.dump(out2, outfile)


if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '12g') \
        .set('spark.executor.memory', '12g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw4')
    parser.add_argument('--input_path', type=str, default='data2')
    parser.add_argument('--n_cluster', type=int, default=10)
    parser.add_argument('--out_file1', type=str, default='out1.json')
    parser.add_argument('--out_file2', type=str, default='out2.json')

    args = parser.parse_args()

    start = time.time()
    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2, sc)
    sc.stop()
    end = time.time()
    print("Runtime:", end-start)