import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='A1T1')
parser.add_argument('--input_file', type=str, default='./backup/data/hw2/nv_data.csv', help='the input file ')
parser.add_argument('--output_file', type=str, default='./backup/data/hw2/alt2.json',
                    help='the output file contains your answers')

parser.add_argument('--k', type=int, default=70, help='filter threshold')
parser.add_argument('--s', type=int, default=50, help='support threshold')

args = parser.parse_args()


import pyspark
import json
import time
from math import ceil
from itertools import combinations
import copy

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc_conf = pyspark.SparkConf() \
    .setAppName('task1') \
    .setMaster('local[*]') \
    .set('spark.driver.memory', '10g') \
    .set('spark.executor.memory', '10g')

sc = pyspark.SparkContext(conf=sc_conf)
sc.setLogLevel("OFF")

num_buckets = 30000000


# debug note: avoids having two lambda functions in one line
def convert_str(string):
    total = sum(ord(char) for char in string)
    return total

def letter_hash(item_list):
    total = sum(map(lambda item: convert_str(item), list(item_list)))
    return total % num_buckets

# find_new_candidates - generalization to 3+ items that takes frequent candidates and generates new candidates to parse through
# debug note: reduces clutter associated with trying to manually fit everything inside a while loop
def find_new_candidates(frequent_candidates):

    if len(frequent_candidates) != 0 and frequent_candidates is not None:
        itemset_size = len(frequent_candidates[0])
        total_size = len(frequent_candidates)
        new_candidates = []

        for i in range(total_size-1):
            for j in range(i+1,total_size):

                if frequent_candidates[i][:-1] == frequent_candidates[j][:-1]:
                    temp_candidates = tuple(sorted(list(set(frequent_candidates[j]).union(set(frequent_candidates[i])))))
                    candidate_tracker = []

                    for subset_item in combinations(temp_candidates, itemset_size):
                        candidate_tracker.append(subset_item)

                    if set(candidate_tracker).issubset(set(frequent_candidates)):
                        new_candidates.append(temp_candidates)
                else:
                    break
        return new_candidates

    return []

# find_candidate_items - the bulk of the work; takes in partition and returns candidates
def find_candidate_items(partition, size, support):
    new_partition = copy.deepcopy(list(partition))
    partition = list(new_partition)
    new_support = ceil(support * len(list(partition)) / size)

    baskets_list = list(partition)
    bitmap = [0 for i in range(num_buckets)]

    counts = {}
    candidates = {}

    for basket in baskets_list:
        # find frequent bucket
        for pair in combinations(basket, 2):
            key = ord(pair[0][0]) + ord(pair[1][0]) % num_buckets
            bitmap[key] += 1

        # find frequent singleton
        for item in basket:

            try:
                counts[item] += 1
            except KeyError:
                counts[item] = 1

    bitmap = list(map(lambda value: True if value >= new_support else False, bitmap))
    filtered_baskets = dict(filter(lambda count: count[1] >= new_support, counts.items()))
    frequent_items = sorted(list(filtered_baskets.keys()))

    # debug note: fixes a bunch of stuff for some reason, not sure why
    possible_candidates = frequent_items

    num_items = 1
    candidates[str(num_items)] = [tuple(item.split(",")) for item in frequent_items]

    # the second phrase, third phrase .... until the candidate list is empty
    while len(possible_candidates) > 0 and possible_candidates is not None:
        num_items += 1
        new_counts = {}

        for basket in baskets_list:
            # Both candidate itemset and high frequency
            basket = sorted(list(set(basket).intersection(set(possible_candidates))))

            # Collecting pairs and higher-order itemsets
            if len(basket) >= num_items:

                # pairs of items
                if num_items == 2:
                    for pair in combinations(basket, num_items):
                        key = ord(pair[0][0]) + ord(pair[1][0]) % num_buckets

                        if bitmap[key]:
                            try:
                                new_counts[pair] += 1

                            except KeyError:
                                new_counts[pair] = 1

                # 3+ groups of items
                if num_items >= 3:
                    for items in new_candidates:

                        # debugging: only update count if items subset of basket
                        if set(items).issubset(set(basket)):
                            try:
                                new_counts[items] += 1

                            except KeyError:
                                new_counts[items] = 1

        candidate_frequencies = dict(filter(lambda count: count[1] >= new_support, new_counts.items()))
        new_candidates = find_new_candidates(sorted(list(candidate_frequencies.keys())))
        if len(candidate_frequencies) == 0:
            break

        candidates[str(num_items)] = list(candidate_frequencies)

    out = []
    for num_items in candidates:
        for item in candidates[num_items]:
            out.append(tuple(sorted([entry for entry in item])))

    return out

# find_frequent_items - final weeding out stage; takes candidates and finds frequent itemsets
def find_frequent_items(partition, candidate_pairs):
    counts = {}

    for pairs in candidate_pairs:
        if set(pairs).issubset(set(partition)):

            try:
                counts[pairs] += 1
            except KeyError:
                counts[pairs] = 1

    yield [tuple((key, value)) for key, value in counts.items()]

# group_lengths groups together output lists for debugging final results
def group_lengths(input_list):
    output_list = [[]]

    for item in input_list:
        try:
            # add item to list of lists
            output_list[len(item)-1].append(item)

        except IndexError:
            # adds empty lists until the correct index is added
            while len(output_list) < len(item):
                output_list.append([])

            output_list[len(item)-1].append(item)
    return output_list


if __name__ == '__main__':
    # start timer
    before_time = time.perf_counter()

    # input csv file and setup output dictionary
    # debug note: splitting up like this avoids calling actions from within transformations or vice versa
    text_rdd = sc.textFile(args.input_file)
    results_dict = {}

    # business_rdd - filtered tuple of business ids
    # Splits lines, groups keys, filters by filter threshold, takes business_id column
    business_rdd = text_rdd.map(lambda line: (line.split(",")[0], line.split(",")[1])) \
        .groupByKey() \
        .filter(lambda kv_pair: kv_pair[0] != "user_id" and kv_pair[1] != "business_id") \
        .map(lambda kv_pair: (kv_pair[0], tuple(set(kv_pair[1])))) \
        .filter(lambda kv_pair: len(kv_pair[1]) > int(args.k)) \
        .map(lambda kv_pair: kv_pair[1]) \
        .persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    # debug note: splitting up like this avoids calling action from within transformation
    size = business_rdd.count()

    # candidate_items - tuple of possible candidates for association rules (first phase)
    # Partitions dataset, finds frequent items in each partition, regroups partitions, sorts pairs
    candidate_items = business_rdd.mapPartitions(
        lambda partition: find_candidate_items(partition, size, int(args.s))) \
        .distinct() \
        .sortBy(lambda item: (len(item), item)) \
        .map(lambda item: tuple([str(entry) for entry in item])) \
        .collect()

    candidate_items_copy = sc.parallelize(candidate_items) \
        .map(lambda item: list([str(entry) for entry in item])) \
        .collect()
    results_dict["Candidates"] = group_lengths(candidate_items_copy)

    # frequent_items - tuple of frequent items for association rules (second phase)
    # Partitions dataset, creates a flat map, groups keys, filters out infrequent baskets, selects baskets, sorts
    frequent_items = business_rdd.flatMap(
        lambda partition: find_frequent_items(partition, candidate_items)) \
        .flatMap(lambda kv_pair: kv_pair) \
        .reduceByKey(lambda w1, w2: w1 + w2) \
        .filter(lambda count: count[1] >= args.s) \
        .map(lambda count: count[0]) \
        .sortBy(lambda kv_pair: (len(list(kv_pair)), tuple(kv_pair))) \
        .collect()

    frequent_items_copy = sc.parallelize(frequent_items) \
        .map(lambda item: [str(entry) for entry in item]) \
        .collect()
    results_dict["Frequent Itemsets"] = group_lengths(frequent_items_copy)

    after_time = time.perf_counter()
    results_dict["Runtime"] = after_time-before_time

    # Write results to json file
    with open(args.output_file, 'w') as output_file:
        json.dump(results_dict, output_file)
    output_file.close()