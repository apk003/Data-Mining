import argparse
import json
import time
import pyspark
import os
import sys


def filter_duplicate(itemset):
    output = {}

    for kv in itemset:
        if kv[0] not in output:
            output[kv[0]] = (0,0)

        output[kv[0]] = (output[kv[0]][0] + kv[1], output[kv[0]][1] + 1)

    for item in output:
        output[item] = output[item][0] / output[item][1]

    return output


def get_combinations(candidates, bus, sim, n):
    output = []

    for candidate in candidates:
        new_bid, new_candidate = bus, candidate

        # symmetric similarities are the same
        if bus > candidate:
            new_bid, new_candidate = candidate, bus

        if (new_bid, new_candidate) in sim:
            output.append((bus, candidate, sim[(new_bid, new_candidate)]))

    return sorted(output, key = lambda kvv: -kvv[2])[:n]


def get_prediction(users_rdd, bus, ratings, sim, n, bus_avg):
    candidates = set(map(lambda kv: kv[0], ratings))
    filtered_ratings = filter_duplicate(ratings)

    output = []
    for candidate in candidates:
        new_bid, new_candidate = bus, candidate

        if bus > candidate:
            new_bid, new_candidate = candidate, bus

        if (new_bid, new_candidate) in sim:
            output.append((bus, candidate, sim[(new_bid, new_candidate)]))

    combinations = sorted(output, key = lambda kvv: -kvv[2])[:n]

    i,j = 0,0
    for kvv in combinations:
        i += filtered_ratings[kvv[1]] * kvv[2]
        j += kvv[2]

    # imputation for missing values
    # replaces missing value with average for user
    # RMSE125 improved by adding if statements for more extreme values
    if i == 0 or j == 0:
        if bus_avg[bus] > 4:
            return (users_rdd, bus, 5)

        if bus_avg[bus] < 3:
            return (users_rdd, bus, 2)

        elif bus_avg[bus] < 2.4:
            return (users_rdd, bus, 1)

        return (users_rdd, bus, round(bus_avg[bus]), 3)

    return (users_rdd, bus, i/j)


# use model file to create predictions
def main(train_file, test_file, model_file, output_file, n_weights, sc):
    key = (0, 0)

    # model_rdd - business input from model file
    # select cols, flatten, get indices
    model_rdd = sc.textFile(model_file) \
        .map(lambda line: [json.loads(line)["b1"], json.loads(line)["b2"]]) \
        .flatMap(lambda kv: kv) \
        .zipWithIndex()
    model_dict = model_rdd.collectAsMap()

    # business_rdd - raw business input from model file (with similarity)
    business_rdd = sc.textFile(model_file) \
        .map(lambda line: ((json.loads(line)["b1"], json.loads(line)["b2"]),
                            json.loads(line)["sim"],))
    business_dict = business_rdd.collectAsMap()

    # train_rdd - raw train file input
    train_rdd = sc.textFile(train_file).cache()

    # users_rdd - raw user input from train file with duplicates combined
    users_rdd = train_rdd.map(lambda line: (json.loads(line)["user_id"],
        [(json.loads(line)["business_id"], json.loads(line)["stars"])])) \
        .reduceByKey(lambda w1,w2: w1 + w2)
    users_dict = users_rdd.collectAsMap()

    # bus_rdd - raw train data turned into dict containing avg rating for business
    # aggregateByKey has poor documentation aside from a stackexchange post
    bus_rdd = sc.textFile(train_file).map(lambda line: (json.loads(line)["business_id"],
        json.loads(line)["stars"])) \
        .aggregateByKey(key, lambda a,b: (a[0] + b, a[1] + 1),
                       lambda a,b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda pair: (pair[0] / pair[1]))
    bus_avg = bus_rdd.collectAsMap()

    # test_rdd test file input for making predictions
    # select cols, filter by existing rating, make prediction, filter for valid predictions
    test_rdd = sc.textFile(test_file) \
        .map(lambda line: (json.loads(line)["user_id"],json.loads(line)["business_id"])) \
        .map(lambda kvv: get_prediction(kvv[0], kvv[1], users_dict[kvv[0]], business_dict, n_weights, bus_avg))
    results = test_rdd.collect()

    # export to json file
    with open(output_file, 'w') as outfile:
        print(len(results))
        for line in results:
            results_dict = {}
            results_dict["user_id"] = line[0]
            results_dict["business_id"] = line[1]
            results_dict["stars"] = line[2]

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
    parser.add_argument('--test_file', type=str, default='./data/test_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--output_file', type=str, default='./outputs/task2.case1.test.out')
    parser.add_argument('--time_file', type=str, default='./outputs/time.out')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))