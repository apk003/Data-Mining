import argparse
import os
import sys
import json
import time
import pyspark
import itertools
import random


# debug note: tuples are faster than lists
def tuple_sort(input_tuple: tuple) -> tuple:

    list_tuple = list(input_tuple)
    list_tuple.sort()

    return tuple(list_tuple)


def sort_pairs(kv_pair):
    kv_pair[1].sort()

    a = kv_pair[0]
    b = kv_pair[1]

    return (a,b)

# debug note: splitting into helper functions improves performance dramatically
def hash_func(a,b,x,m):
    return (a*x + b) % 40009 % m

def hash_primes(length):
    params = []

    i = 0
    while i < length:
        i += 1

        # get parameters for hashing
        a = random.randint(1,30000000)
        b = random.randint(1,30000000)
        params.append((a,b))

    return params

def min_hash(vals, parameters, signature_length):
    result = []
    for (a,b) in parameters:
        signatures = []

        # apply hash function to each item
        for val in vals:
            signatures.append(hash_func(a,b,val,signature_length))
        result.append(min(signatures))

    return result

# debug note: split and hash signatures here instead of directly after minhash
def get_hash(val,row,bus):
    result = []
    vals = []

    band = 0
    i = 0
    # hashes signatures
    while(i < len(val)):

        vals.append(val[i])
        i += 1

        if i % row == 0:
            result.append(((hash(tuple(vals)),band),[bus]))

            vals = []
            band += 1
    return result


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

 # find intersection / union
    shared_data = set1.intersection(set2)
    total = set1.union(set2)

    return (len(shared_data) / len(total))

def check_sim(kv, utility_matrix):
    bus1 = kv[0]
    bus2 = kv[1]

    idx1 = utility_matrix[bus1]
    idx2 = utility_matrix[bus2]

    sim = jaccard_similarity(idx1, idx2)
    return ((bus1, bus2), sim)


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc):

    # text_rdd - raw input file
    text_rdd = sc.textFile(input_file) \
        .cache()

    # users_rdd - dictionary containing users and their indices
    users_rdd = text_rdd.map(lambda line: json.loads(line)['user_id']) \
        .distinct() \
        .zipWithIndex() \
        .collectAsMap()

    # pairs_rdd - key-value pairs of users and the businesses they have reviewed
    pairs_rdd = text_rdd.map(lambda line: (json.loads(line)['business_id'], users_rdd[json.loads(line)['user_id']])) \
        .distinct() \
        .map(lambda line: (line[0], list([line[1]])))\
        .reduceByKey(lambda w1,w2: w1 + w2) \
        .map(lambda line: sort_pairs(line))
    pairs_dict = pairs_rdd.collectAsMap()

    # params - a,b,m for hash function
    # signatures - hashed signatures / utility matrix
    params = hash_primes(n_bands*n_rows)
    signatures = pairs_rdd.mapValues(lambda uid: min_hash(uid, params, len(users_rdd)-1))

    # candidate_rdd - the bulk of computation; uses jaccard sim to find similar pairs
    candidate_rdd = signatures \
        .flatMap(lambda kv: get_hash(kv[1], n_rows, kv[0])) \
        .reduceByKey(lambda w1,w2: w1 + w2) \
        .filter(lambda kv: len(kv[1]) > 1) \
        .flatMap(lambda kv: list(itertools.combinations(kv[1], 2))) \
        .map(lambda kv: tuple_sort(kv)).distinct() \
        .map(lambda kv: check_sim(kv, pairs_dict)) \
        .filter(lambda item: item[1] >= jac_thr)
    candidate_list = candidate_rdd.collect()

    # export to json file
    with open(output_file, 'w') as outfile:
        for line in candidate_list:
            results_dict = {}
            results_dict['b1'] = line[0][0]
            results_dict['b2'] = line[0][1]
            results_dict['sim'] = line[1]

            # puts each similar pair on a new line
            print(json.dumps(results_dict), file=outfile)


if __name__ == '__main__':
    start_time = time.time()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_file', type=str, default='text_rdd.json')
    parser.add_argument('--output_file', type=str, default='task1.out')
    parser.add_argument('--time_file', type=str, default='task1.time')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--n_bands', type=int, default=50)
    parser.add_argument('--n_rows', type=int, default=2)
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold, args.n_bands, args.n_rows, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))

