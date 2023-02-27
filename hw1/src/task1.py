import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='A1T1')
parser.add_argument('--input_file', type=str, default='./backup/data/hw1/review.json', help='the input file ')
parser.add_argument('--output_file', type=str, default='./backup/data/hw1/alt1.json',
                    help='the output file contains your answers')
parser.add_argument('--stopwords', type=str, default='./backup/data/hw1/stopwords',
                    help='the file contains the stopwords')
parser.add_argument('--y', type=int, default=2018, help='year')
parser.add_argument('--m', type=int, default=10, help='top m users')
parser.add_argument('--n', type=int, default=10, help='top n frequent words')

args = parser.parse_args()


import pyspark
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == '__main__':

    sc_conf = pyspark.SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '10g') \
        .set('spark.executor.memory', '10g')


    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

input_file = args.input_file
text_rdd = sc.textFile(input_file,8).map(lambda line: json.loads(line))

with open(args.stopwords, 'r') as stopwords_file:
    stopwords = stopwords_file.read().split("\n")

results = {}

# total_reviews: total number of reviews; simply counts distinct reviews
total_reviews = text_rdd.map(lambda row: (row["review_id"])).count()
results['A'] = total_reviews

# reviews_by_year: total number of reviews in year y
# Filter by year, select review_id, count reviews
reviews_by_year = text_rdd.filter(lambda row: (row["date"][0:4] == str(args.y))) \
    .map(lambda row: row["review_id"]).count()
results['B'] = reviews_by_year

# unique_users: total number of distinct user_id values
unique_users = text_rdd.map(lambda row: row["user_id"]).distinct().count()
results['C'] = unique_users

# top_users: list of key-value pairs containing m most active users
# Select user_id and business_id (no spamming one business), count instances, combine keys, sort
top_users = text_rdd.map(lambda row: (row['business_id'], row['user_id'])) \
    .map(lambda user: (user[1],1)) \
    .reduceByKey(lambda w1,w2: w1+w2) \
    .takeOrdered(args.m, lambda key: (-key[1], key[0]))
results['D'] = top_users

# word_counts: list containing n most common words
# Select text column, split and strip words, count instances, remove stopwords, combine keys, sort, select word column
word_counts = text_rdd.map(lambda row: (row["text"])) \
    .flatMap(lambda line: line.lower().split(" ")) \
    .map(lambda word: (word.strip(".,?!/\\\"\'\n\t()[];:{}~`-_<>@#$%&^*1234567890"),1)) \
    .filter(lambda word: word[0] not in stopwords and word[0] != '' and len(word[0].split('\'')) == 1) \
    .reduceByKey(lambda w1,w2: w1+w2) \
    .takeOrdered(args.n, key=lambda key: -key[1])
results['E'] = list(map(lambda word: word[0], word_counts))

# Write results to json file
with open(args.output_file, 'w') as output_file:
    json.dump(results, output_file)
output_file.close()

