import argparse
import pyspark
import graphframes
import time
import os
import sys


def filter_by_thr(pair, pairs_dict, filter_thr):
    bus1 = pairs_dict[pair[0]]
    bus2 = pairs_dict[pair[1]]

    intersect = bus1.intersection(bus2)

    return len(intersect) >= filter_thr

def main(filter_threshold, input_file, output_file, sc):

    input_rdd = sc.textFile(input_file)
    header = "user_id,business_id"

    text_rdd = input_rdd.filter(lambda line: line != header) \
        .map(lambda line: (line.split(",")[0], line.split(",")[1])) \
        .cache()

    users_rdd = text_rdd.map(lambda item: item[0]) \
        .distinct().zipWithIndex()
    users_dict = users_rdd.collectAsMap()
    total_users = len(users_dict)

    users_idx = users_rdd.map(lambda pair: (pair[1], pair[0]))
    users_index_dict = users_idx.collectAsMap()

    bus_rdd = text_rdd.map(lambda item: item[1]) \
        .distinct().zipWithIndex()
    bus_dict = bus_rdd.collectAsMap()

    pairs_rdd = text_rdd.map(lambda x: (users_dict[x[0]], bus_dict[x[1]])) \
        .groupByKey().mapValues(set)
    pairs_dict = pairs_rdd.collectAsMap()

    candidates_rdd = sc.parallelize(i for i in range(total_users)) \
        .flatMap(lambda i: [(i,j) for j in range(i+1, total_users)]) \
        .filter(lambda pair: filter_by_thr(pair, pairs_dict, filter_threshold))

    nodes_rdd = candidates_rdd.flatMap(lambda pair: pair) \
        .distinct() \
        .map(lambda pair: (users_index_dict[pair],))
    nodes_df = nodes_rdd.toDF(["id"])

    edges_rdd = candidates_rdd.flatMap(lambda pair: (pair, (pair[1], pair[0]))) \
        .map(lambda kvv: (users_index_dict[kvv[0]], users_index_dict[kvv[1]]))
    edges_df = edges_rdd.toDF(["src", "dst"])

    g = graphframes.GraphFrame(nodes_df, edges_df)
    results_df = g.labelPropagation(maxIter=5)

    results_rdd = results_df.rdd.map(lambda res: [res[1], res[0]]) \
        .groupByKey() \
        .map(lambda pair: sorted(pair[1])) \
        .sortBy(lambda item: (len(item), item[0]))
    results = results_rdd.collect()

    with open(output_file, 'w') as outfile:
        for line in results:

            out = ""
            for user in sorted(line):
                out += "\'" + str(user) + "\', "

            out = out[:-2]
            print(out, file=outfile)


if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    sqlContext = pyspark.sql.SparkSession.builder \
        .appName("Task1") \
        .master("local[*]") \
        .getOrCreate()

    parser = argparse.ArgumentParser(description='hw4')
    parser.add_argument('--filter_threshold', type=int, default=7)
    parser.add_argument('--input_file', type=str, default='ub_sample_data.csv')
    parser.add_argument('--community_output_file', type=str, default='task1out.txt')

    args = parser.parse_args()

    start = time.time()
    main(args.filter_threshold, args.input_file, args.community_output_file, sc)
    sc.stop()
    sqlContext.stop()
    end = time.time()

    print("Runtime:", end-start)
