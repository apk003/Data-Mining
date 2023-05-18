import pyspark
import sys
import os
import argparse
import math
import json
import numpy as np
import xgboost


def avg(iter):
    total = 0

    for item in iter:
        total += float(item[1])

    return (total / len(iter))


# pearson correlation between two businesses for CF
def pearson_cor(bus1, bus2):
    co_rated = list(set(bus1.keys()) & set(bus2.keys()))

    # running lists taken from co_rated
    co_rated1 = []
    co_rated2 = []

    for user in co_rated:
        co_rated1.append(float(bus1[user]))
        co_rated2.append(float(bus2[user]))

    avg1 = sum(co_rated1) / len(co_rated1)
    avg2 = sum(co_rated2) / len(co_rated2)

    sum1 = 0
    sum2 = 0
    total = 0

    for real1, real2 in zip(co_rated1, co_rated2):
        total += ((real1 - avg1) * (real2 - avg2))
        sum1 += ((real1 - avg1) * (real1 - avg1))
        sum2 += ((real2 - avg2) * (real2 - avg2))

    # undefined denominator
    if sum1 * sum2 == 0:
        return 0

    return total / (math.sqrt(sum1) * math.sqrt(sum2))


# generate predictions
def predict(test_train, pairs_dict, bus_avg):
    bus = test_train[0]
    neighbors = list(test_train[1])
    vals = []

    for val in neighbors:
        key = (val[0], bus)
        vals.append((float(val[1]), pairs_dict.get(key, 0)))

    # debug note: taking a smaller subset drastically reduces memory usage!
    top40 = sorted(vals, key=lambda val: val[1], reverse=True)[:40]
    numerator = 0
    denominator = 0

    for val in top40:
        numerator += (val[0] * val[1])
        denominator += abs(val[1])

    # impute average business rating
    if denominator * numerator == 0:
        avg = bus_avg.get(bus)
        return [bus, avg, len(neighbors)]

    return [bus, numerator / denominator, len(neighbors)]


# collaborative filtering (CF) model for recommendation system
def CF(train_data, test_data):
    # users_rdd - users and indices
    users_rdd = train_data.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex()
    users_dict = users_rdd.collectAsMap()

    # bus_rdd - businesses and indices
    bus_rdd = train_data.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex()
    bus_dict = bus_rdd.collectAsMap()

    # reversed versions of above
    user_index = {idx: user for user, idx in users_dict.items()}
    bus_index = {idx: business for business, idx in bus_dict.items()}

    # bus_user_rdd - businesses as keys, (user, rating) as values
    bus_user_rdd = train_data.map(lambda x: (bus_dict[x[1]], (users_dict[x[0]], x[2]))) \
        .groupByKey().mapValues(list)
    bus_user_dict = bus_user_rdd.collectAsMap()

    # user_bus_rdd - users as keys, (business, rating) as values
    user_bus_rdd = train_data.map(lambda x: (users_dict[x[0]], (bus_dict[x[1]], x[2]))) \
        .groupByKey().mapValues(list)
    user_bus_dict = user_bus_rdd.collectAsMap()

    # averages for each set
    bus_avg = bus_user_rdd.map(lambda x: (x[0], avg(x[1]))).collectAsMap()
    user_avg = user_bus_rdd.map(lambda x: (x[0], avg(x[1]))).collectAsMap()

    # preprocessed testing data
    test_user_business_rdd = test_data.map(lambda x: (users_dict.get(x[0], -1), bus_dict.get(x[1], -1))) \
        .filter(lambda x: x[0] != -1 and x[1] != -1)

    # filtered testing data
    filtered_pairs = test_data.filter(
        lambda pair: pair[0] not in users_dict.keys() or pair[1] not in bus_dict.keys()).collect()

    # total_rdd - join test data with pairs to generate all possible candidates
    total_rdd = test_user_business_rdd.leftOuterJoin(user_bus_rdd)

    # pairs_rdd - all possible candidates filtered down to important pairs
    pairs_rdd = total_rdd.flatMap(lambda x: [(val[0], x[1][0]) for val in x[1][1]]) \
        .filter(lambda pair: len(set(dict(bus_user_dict.get(pair[0])).keys()) & set(dict(bus_user_dict.get(pair[1])).keys())) >= 175) \
        .map(lambda pair: (pair, pearson_cor(dict(bus_user_dict.get(pair[0])), dict(bus_user_dict.get(pair[1]))))) \
        .filter(lambda pair: pair[1] > 0).map(lambda pair: {(pair[0][0], pair[0][1]): pair[1]}) \
        .flatMap(lambda pair: pair.items())
    pairs_dict = pairs_rdd.collectAsMap()

    # result dictionaries
    predictions = total_rdd.map(lambda x: (x[0], predict(x[1], pairs_dict, bus_avg)))
    out_dict = predictions.map(lambda pair: ((user_index[pair[0]], bus_index[pair[1][0]]), pair[1][1])).collect()
    neighbor_preds = predictions.map(lambda pair: ((user_index[pair[0]], bus_index[pair[1][0]]), pair[1][2])).collectAsMap()

    # imputation step
    for pair in filtered_pairs:
        # impute user average for unknown business
        if pair[0] in users_dict.keys():
            out_dict.append((tuple(pair), user_avg[users_dict[pair[0]]]))
            neighbor_preds[tuple(pair)] = len(user_bus_dict[users_dict[pair[0]]])

        # impute business average for unknown user
        elif pair[1] in bus_dict.keys():
            out_dict.append((tuple(pair), bus_avg[bus_dict[pair[0]]]))
            neighbor_preds[tuple(pair)] = 0

        # impute 2.5 for unknown pair
        else:
            out_dict.append((tuple(pair), 2.5))
            neighbor_preds[tuple(pair)] = 0

    return out_dict, neighbor_preds


# model-based training for recommendation system
def model_based(train_data, test_data, user_file, bus_file):

    # select users from training data
    users = train_data.map(lambda pair: pair[0])
    users_train = set(users.distinct().collect())

    # select businesses from training data
    bus = train_data.map(lambda pair: pair[1])
    bus_train = set(bus.distinct().collect())

    # users_rdd - raw user data from user file
    users_rdd = sc.textFile(user_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x["user_id"] in users_train) \
        .map(lambda x: (x["user_id"], (x["review_count"], x["average_stars"])))
    users_dict = users_rdd.collectAsMap()

    # bus_rdd - raw business data from business file
    bus_rdd = sc.textFile(bus_file).map(lambda x: json.loads(x)) \
        .filter(lambda x: x["business_id"] in bus_train) \
        .map(lambda x: (x["business_id"], (x["review_count"], x["stars"])))
    bus_dict = bus_rdd.collectAsMap()

    # numpy preprocessing
    X_rdd = train_data.map(lambda X: np.array([users_dict[X[0]], bus_dict[X[1]]]).flatten()).collect()
    train_X = np.array(X_rdd)
    y_rdd = train_data.map(lambda y: float(y[2])).collect()
    train_y = np.array(y_rdd)

    # test set
    test_X = np.array(test_data.map(lambda x: np.array([users_dict.get(x[0], [0, 2.5]), bus_dict.get(x[1], [0, 2.5])]).flatten())\
            .collect())

    # model - gradient boost regressor of which the model is based on
    model = xgboost.XGBRegressor(n_estimators=95,
                                 max_depth=10,
                                 learning_rate=0.095,
                                 objective="reg:squarederror",
                                 booster="gbtree",
                                 min_child_weight=1,
                                 colsample_bytree=1,
                                 reg_lambda=1,
                                 subsample=1,
                                 reg_alpha=0,
                                 gamma=0)

    # debugging hyperparameter notes:
    # max_depth should stay around 10
    # n_estimators should be 1000*learning_rate
    # learning rate under 0.1
    # min_child_weight through gamma are less meaningful

    model.fit(train_X, train_y)
    prediction = model.predict(test_X)

    results = []
    test_data = test_data.collect()

    # add results to list
    for pair in zip(test_data, prediction):
        results.append(((pair[0][0], pair[0][1]), pair[1]))

    return results

# rounds data to improve rmse125
def rmse125(data):
    if data > 4.5:
        return 5.0

    elif data < 2.5:
        return 2.0

    elif data < 1.5:
        return 1.0

    else:
        return data

# Hybrid Recommendation System using CF (item and user based) and MB (XGBRegressor) algorithms
def main(train_file, user_file, bus_file, test_file, output_file, sc):
    # bus_metadata: raw data to be used for improvement of model (maybe in the future?)
    bus_metadata = sc.textFile(train_file).map(lambda line: json.loads(line)) \
        .map(lambda line: (line["business_id"], line["stars"], line["useful"], line["funny"], line["cool"])) \
        .collect()

    # train_data - raw data from train file
    train_data = sc.textFile(train_file).map(lambda line: json.loads(line)) \
        .map(lambda line: (line["user_id"], line["business_id"], line["stars"])) \

    # test_data - raw data from test file
    test_data = sc.textFile(test_file).map(lambda line: json.loads(line)) \
        .map(lambda line: (line["user_id"], line["business_id"])) \

    # run model-based and CF predictions
    mb_val = model_based(train_data, test_data, user_file, bus_file)
    cf_val, neighbor_val = CF(train_data, test_data)
    neighbor_max = max(neighbor_val.values())

    mb_normal = []
    cf_normal = []

    # normalize MB results
    for pair in mb_val:
        mb_normal.append((pair[0], (1 - float(neighbor_val[pair[0]] / neighbor_max)) * pair[1]))

    # normalize CF results
    for pair in cf_val:
        cf_normal.append((pair[0], float(neighbor_val[pair[0]] / neighbor_max) * pair[1]))

    # final result is some weighted average of CF and MB results
    # to improve rmse125, the end result goes through a custom rounding procedure
    mb_cf = mb_normal + cf_normal
    results = sc.parallelize(mb_cf).reduceByKey(lambda w1,w2: w1 + w2) \
        .mapValues(rmse125) \
        .collect()

    # export to json file
    with open(output_file, 'w') as outfile:
        print(len(results))
        for line in results:
            results_dict = {}
            results_dict["user_id"] = line[0][0]
            results_dict["business_id"] = line[0][1]
            results_dict["stars"] = line[1]

            # puts each result on a new line
            print(json.dumps(results_dict), file=outfile)


if __name__ == "__main__":

    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('competiton_predict') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '12g') \
        .set('spark.executor.memory', '12g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='competition')
    parser.add_argument('--train_file', type=str, default="train_review.json")
    parser.add_argument('--user_file', type=str, default="user.json")
    parser.add_argument('--bus_file', type=str, default="business.json")
    parser.add_argument('--test_file', type=str, default="test_review.json")
    parser.add_argument('--output_file', type=str, default="out.json")

    args = parser.parse_args()

    main(args.train_file, args.user_file, args.bus_file, args.test_file, args.output_file, sc)
    sc.stop()