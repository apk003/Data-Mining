import argparse
import json
import time
import pyspark
import os
import sys
import math


def filter_co_rated_thr(user1, user2, pairs_dict, co_rated_thr):
    user_set1 = set(pairs_dict[user1].keys())
    user_set2 = set(pairs_dict[user2].keys())

    #intersection = user_set1.intersection(user_set2)
    #print(len(intersection), len(user_set1 & user_set2))
    return len(user_set1 & user_set2) >= co_rated_thr

def get_avg(dict, corated):
    total = 0

    for user in corated:
        total += dict[user]

    return total / len(corated)

def normalize(dict, corated):
    avg = get_avg(dict, corated)

    out = {}
    for one in corated:
        out[one] = dict[one] - avg

    return out

def calculate_pearson(bus_id1, bus_id2, pairs_dict):
    bus1 = pairs_dict[bus_id1]
    bus2 = pairs_dict[bus_id2]

    corated = set(bus1.keys()).intersection(set(bus2.keys()))

    bus1 = normalize(bus1, corated)
    bus2 = normalize(bus2, corated)

    weights, total1, total2 = 0, 0, 0
    for item in corated:
        weights += bus1[item]*bus2[item]
        total1 += (bus1[item])**2
        total2 += (bus2[item])**2

    distance_from_zero = math.sqrt(total1 * total2)

    if distance_from_zero == 0:
        return 0
    return weights / distance_from_zero


def main(train_file, model_file, co_rated_thr, sc):
    start = time.time()
    text_rdd = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"], x["stars"])).cache()

    # users_rdd - users with index
    users_rdd = text_rdd.map(lambda x: x[0]) \
        .distinct() \
        .zipWithIndex() \

    # business_rdd - businesses with index
    business_rdd = text_rdd.map(lambda x: x[1]) \
        .distinct() \
        .zipWithIndex() \

    business_dict = business_rdd.map(lambda x: (x[1], x[0])).collectAsMap()
    num_business = business_rdd.count()

    # kvv_rdd - user, business, avg ratings
    # select rows, join and select, reduce, find averages, clean up
    kvv_rdd = text_rdd.map(lambda line: (line[1], (line[0], line[2]))) \
        .join(business_rdd).map(lambda kvv: (kvv[1][0][0], (kvv[1][1], kvv[1][0][1]))) \
        .join(users_rdd).map(lambda kvv: ((kvv[1][0][0], kvv[1][1]), kvv[1][0][1])) \
        .groupByKey().mapValues(list) \
        .mapValues(lambda item: sum(item) / len(item)) \
        .map(lambda bus: (bus[0][0], (bus[0][1], bus[1]))) \
        .groupByKey().mapValues(dict)
    kvv_dict = kvv_rdd.collectAsMap()

    # pairs_rdd - all possible candidate pairs filtered by co_rated_thr and pearson
    pairs_rdd = sc.parallelize([i for i in range(num_business)]) \
        .flatMap(lambda i: [(i, j) for j in range(i + 1, num_business)]) \
        .filter(lambda pair: filter_co_rated_thr(pair[0], pair[1], kvv_dict, co_rated_thr)) \
        .map(lambda x: (x[0], x[1], calculate_pearson(x[0], x[1], kvv_dict))) \
        .filter(lambda x: x[2] > 0.1) \
        .distinct() \
        .filter(lambda line: line is not None)
    results = pairs_rdd.collect()

    # export to json file
    with open(model_file, 'w') as outfile:
        for line in results:
            results_dict = {}
            results_dict["b1"] = business_dict[line[0]]
            results_dict["b2"] = business_dict[line[1]]
            results_dict["sim"] = line[2]

            # puts each similar pair on a new line
            print(json.dumps(results_dict), file=outfile)



if __name__ == '__main__':
    start_time = time.time()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--time_file', type=str, default='./outputs/time.out')
    parser.add_argument('--m', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.m, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))

    # debug code



