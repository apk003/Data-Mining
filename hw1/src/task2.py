import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='A1T1')
parser.add_argument('--review_file', type=str, default='./backup/data/hw1/review.json',
                    help='the reviews file')
parser.add_argument('--business_file', type=str, default='./backup/data/hw1/business.json',
                    help='the business file')
parser.add_argument('--output_file', type=str, default='./backup/data/hw1/alt2.json',
                    help='contains output of task')
parser.add_argument('--n', type=int, default=10, help='top n categories')

args = parser.parse_args()

import pyspark
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '10g') \
        .set('spark.executor.memory', '10g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

review_rdd = sc.textFile(args.review_file, 8).map(lambda line: json.loads(line))
business_rdd = sc.textFile(args.business_file, 8).map(lambda line: json.loads(line))

# total_stars_rdd: tuple of business_id, total stars, and number of businesses
# Selects business_id and stars cols, group duplicate keys, emit (id,stars,len)
total_stars_rdd = review_rdd.map(lambda row: [row["business_id"], row["stars"]]) \
    .groupByKey() \
    .map(lambda x: [x[0], [sum(x[1]), len(x[1])]])

# categories_rdd: lists of categories for each business (tuple doesn't work for list comprehension)
# Selects business_id and categories cols, filters out null values, split and strip categories
categories_rdd = business_rdd.map(lambda row: (row["business_id"], row["categories"])) \
    .filter(lambda x: (x[1] not in ("", None))) \
    .mapValues(lambda values: [value.strip() for value in values.split(',')])

# join_rdd: joined RDD separate from avg_stars_rdd to avoid memory issues
join_rdd = categories_rdd.leftOuterJoin(total_stars_rdd)

# avg_stars_rdd: tuples of average stars for each category
# Emits key-value pairs, filters out null values (debug purposes)
# Emit (category, stars), add like keys, divide total stars by num businesses, sort by stars
avg_stars_rdd = join_rdd.map(lambda key: key[1]) \
    .filter(lambda key: key[1] is not None) \
    .flatMap(lambda key: ((category, key[1]) for category in key[0])) \
    .reduceByKey(lambda w1,w2: (w1[0] + w2[0], w1[1] + w2[1])) \
    .mapValues(lambda x: (x[0] / x[1])) \
    .takeOrdered(args.n, lambda key: [-key[1], key[0]])

results_dict = {"result": []}
results_dict["result"] = avg_stars_rdd

# Write results to json file
with open(args.output_file, 'w') as output_file:
    json.dump(results_dict, output_file)
output_file.close()
