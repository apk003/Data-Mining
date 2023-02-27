import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='A1T1')
parser.add_argument('--input_file', type=str, default='./backup/data/hw1/review.json',
                    help='the reviews file')
parser.add_argument('--output_file', type=str, default='./backup/data/hw1/alt3_customized.json',
                    help='contains output of task')
parser.add_argument('--n_partitions', type=int, default=27, help='number of partitions')
parser.add_argument('--n', type=str, default=4.9,
                    help='stars threshold')

args = parser.parse_args()

import pyspark
import json
import random

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('task3_customized') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '10g') \
        .set('spark.executor.memory', '10g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

# custom_partitioner: impractical partitioner that takes a key and returns a random number up to 200
def custom_partitioner(key):
        return random.randint(0,200)

# Testing runtimes
import time
before_time = time.perf_counter()

# business_rdd:
business_rdd = sc.textFile(args.input_file, 8).map(lambda line: json.loads(line)) \
    .map(lambda row: (row["business_id"],1)) \
    .partitionBy(int(args.n_partitions), custom_partitioner)

results_dict = {}

# n_partitions:
n_partitions = args.n_partitions
results_dict["n_partitions"] = n_partitions

# n_items:
n_items = business_rdd.glom().map(len).collect()
results_dict["n_items"] = n_items

# result:
result = business_rdd.reduceByKey(lambda w1,w2: w1+w2) \
    .filter(lambda x: float(args.n) < x[1]).collect()
results_dict["result"] = result

with open(args.output_file, 'w') as output_file:
    json.dump(results_dict, output_file)
output_file.close()

# some data: 27 in 57 seconds, 30 in 60, 35 in 66, 15 in 38, 20 in 46, 5 in 23, 1 in 18
# choosing keys at random is a better partitioning strategy than the default
after_time = time.perf_counter()
print("Runtime:",after_time-before_time)