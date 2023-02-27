import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='A1T1')
parser.add_argument('--input_file', type=str, default='./backup/data/hw1/review.json',
                    help='the reviews file')
parser.add_argument('--output_file', type=str, default='./backup/data/hw1/alt3_default.json',
                    help='contains output of task')
parser.add_argument('--n', type=str, default=4,
                    help='stars threshold')

args = parser.parse_args()

import pyspark
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('task3_default') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '10g') \
        .set('spark.executor.memory', '10g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

# Testing runtimes
import time
before_time = time.perf_counter()

# business_rdd:
business_rdd = sc.textFile(args.input_file, 8).map(lambda line: json.loads(line)) \
    .map(lambda row: (row["business_id"],1)) \
    .partitionBy(27)

results_dict = {}

# n_partitions:
n_partitions = business_rdd.getNumPartitions()
results_dict["n_partitions"] = n_partitions

# n_items:
n_items = business_rdd.glom().map(len).collect()
results_dict["n_items"] = n_items

# result:
result = business_rdd.reduceByKey(lambda w1,w2: w1+w2) \
    .filter(lambda x: int(args.n) < x[1]).collect()
results_dict["result"] = result

with open(args.output_file, 'w') as output_file:
    json.dump(results_dict, output_file)
output_file.close()

# runs for 42 seconds regardless of partitions
after_time = time.perf_counter()
print("Runtime:",after_time-before_time)