import argparse
import pyspark
import time
import os
import sys


class Node:
    def __init__(self, id):
        self.nodes = 1
        self.edges = 0
        self.name = id
        self.level = 0
        self.children = {}
        self.parent = {}

    def __repr__(self):
        return f"[name: {self.name}, level: {self.level}, parent: {self.parent.keys()}, children: {self.children.keys()}]"


def create_tree(graph, id):
    root = Node(id)

    # initialize with root node
    tree = {0:set()}
    tree[0].add(root)
    added_to_tree = {id: root}

    # runs until queue is empty
    node_queue = [id]
    while node_queue:
        current_node = node_queue.pop(0)

        # traverse all neighbors
        for neighbor in graph[current_node]:
            if neighbor not in added_to_tree.keys():
                new_node = Node(neighbor)

                # initialize neighbor nodes
                new_node.level = added_to_tree[current_node].level + 1
                new_node.parent[current_node] = added_to_tree[current_node]
                new_node.nodes = added_to_tree[current_node].nodes

                try:
                    tree[new_node.level].add(new_node)

                except KeyError:
                    tree[new_node.level] = {new_node}

                # updates dict keeping track of all nodes
                added_to_tree[current_node].children[neighbor] = new_node
                added_to_tree[neighbor] = new_node

                node_queue.append(neighbor)

            else:
                neighbor_node = added_to_tree[neighbor]

                # update existing entries if necessary
                if neighbor_node.level > added_to_tree[current_node].level:
                    neighbor_node.nodes += added_to_tree[current_node].nodes

                    added_to_tree[current_node].children[neighbor] = neighbor_node
                    neighbor_node.parent[current_node] = added_to_tree[current_node]

    # completed tree
    return tree


def get_betweenness(graph_dict):
    out = {}

    # traverse all nodes and add to out dict
    # turn graph into tree
    for node_id in graph_dict.keys():
        tree = create_tree(graph_dict, node_id)

        # traverse all levels of tree
        tree_height = max(tree.keys())
        for level in range(tree_height, 0, -1):

            # betweenness sum for each edge
            for node in tree[level]:
                edge_sum = node.edges + 1

                # work up through parent nodes
                for parent_id, parent_node in node.parent.items():
                    between_edges = (edge_sum / node.nodes) * parent_node.nodes
                    parent_node.edges += between_edges

                    # debug note: algorithm works best using sets, but regular sets are unhashable!
                    edge = frozenset({node.name, parent_id})

                    try:
                        out[edge] += between_edges

                    except KeyError:
                        out[edge] = between_edges

    # everything is counted twice
    for item in out:
        out[item] = out[item] / 2
    return out


def filter_by_thr(pair, filter_thr, pairs_dict):
    bus1 = set(pairs_dict[pair[0]])
    bus2 = set(pairs_dict[pair[1]])

    intersect = bus1.intersection(bus2)
    return len(intersect) >= filter_thr


def remove_edges(graph, removed_edges):
    for edge in removed_edges:
        node1, node2 = list(edge)[0], list(edge)[1]

        graph[node1].remove(node2)
        graph[node2].remove(node1)

    return graph


def get_communities(graph_dict, degree_dict, edges):

    max_communities, max_modularity = get_modularity(graph_dict, degree_dict, edges)
    edges_queue = edges.copy()
    while (len(edges_queue) > 0):
        between_edges = get_betweenness(graph_dict)

        # Bulk of Girvan-Newman algorithm
        # Remove edges with high betweenness

        removed_edges = set()
        if len(between_edges) < 2:
            removed_edges = set(between_edges.keys())

        between_edges = sorted(list(between_edges.items()), key=lambda edge: -edge[1])
        removed_edges = {between_edges[0][0]}
        max_between = between_edges[0][1]

        for edge in between_edges:
            if edge[1] == max_between:
                removed_edges.add(edge[0])

        # remove edges of highest betweenness
        graph_dict = remove_edges(graph_dict, removed_edges)
        for edge in removed_edges:
            edges_queue.remove(edge)

        communities, modularity = get_modularity(graph_dict, degree_dict, edges)
        if modularity > max_modularity:
            max_modularity, max_communities = modularity, communities

    # most probable community sets
    return max_communities


def get_modularity(graph_dict, degree, edges):
    m2 = sum(degree.values())  # 2m, i.e. doulbe of num edges in original graph
    count = 0

    # Modularity step of Girvan-Newman algorithm
    # Modularity implemented using queue system for all nodes

    communities = []
    node_visited = set()
    for node in graph_dict.keys():
        if node not in node_visited:

            # initialize node
            visited = {node}
            queue = [node]
            while queue:

                # grab from queue and search neighbors
                node = queue.pop(0)
                for neighbor in graph_dict[node]:

                    # add neighbors to queue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            community = visited

            communities.append(community)
            node_visited = node_visited.union(community)

    # calculate k_i and k_j for all members of communities
    for community in communities:
        for i in community:
            k_i = degree[i]

            for j in community:
                k_j = degree[j]
                count -= k_i * k_j / m2

                # running total increments each time pair is repeated
                if frozenset({i, j}) in edges:
                    count += 1

    return communities, (count / m2)


def main(filter_threshold, input_file, betweenness_file, community_file, sc):

    header = "user_id,business_id"

    # text_rdd - raw text file as RDD with header filtered out
    text_rdd = sc.textFile(input_file) \
        .filter(lambda line: line is not header) \
        .map(lambda line: line.split(",")) \
        .cache()

    # users_rdd - users column with attached indices
    users_rdd = text_rdd.map(lambda line: line[0]) \
        .distinct() \
        .zipWithIndex()
    users_dict = users_rdd.collectAsMap()
    total_users = len(users_dict)

    users_index_dict = users_rdd.map(lambda user: (user[1], user[0])) \
        .collectAsMap()

    # business_rdd - businesses column with attached indices
    business_rdd = text_rdd.map(lambda bus: bus[1]) \
        .distinct() \
        .zipWithIndex()
    business_dict = business_rdd.collectAsMap()
    total_bus = len(business_dict)

    # pairs_rdd - user,business key-value pairs with businesses grouped by user
    pairs_rdd = text_rdd.map(lambda line: (users_dict[line[0]], business_dict[line[1]])) \
        .groupByKey() \
        .mapValues(lambda val: set(val))
    pairs_dict = pairs_rdd.collectAsMap()

    # candidate_edge_rdd - all possible edges followed by filtering
    # creates all edges, filters by threshold, groups businesses and users by pair
    candidate_edge_rdd = sc.parallelize([i for i in range(total_users)]) \
        .flatMap(lambda i: [(i, j) for j in range(i + 1, total_users)]) \
        .filter(lambda pair: filter_by_thr(pair, filter_threshold, pairs_dict)) \
        .flatMap(lambda pair: [pair, (pair[1], pair[0])]) \
        .groupByKey() \
        .mapValues(lambda val: set(val))
    graph_dict = candidate_edge_rdd.collectAsMap()

    # degrees_dict - mapping of all nodes to their degrees
    degrees_dict = {}
    for node, neighbors in graph_dict.items():
        degrees_dict[node] = len(neighbors)

    # betweenness_dict - mapping of all edges to their betweenness
    betweenness_dict = get_betweenness(graph_dict)

    between_edges = {}
    for edge, betweenness_score in betweenness_dict.items():
        between_edges[tuple(sorted(map(lambda user: users_index_dict[user], edge)))] = betweenness_score

    communities = get_communities(graph_dict, degrees_dict, set(betweenness_dict.keys()))
    communities = map(lambda community: [users_index_dict[user] for user in community], communities)
    communities = list(map(lambda community: (sorted(community), len(community)), communities))

    with open(betweenness_file, 'w') as outfile:
        for pair in sorted(list(between_edges.items()), key=lambda item: (-item[1], item[0])):
            print(str(pair)[1: -1], file=outfile)

    with open(community_file, 'w') as outfile:
        for line in sorted(communities, key=lambda item: (item[1], item[0])):
            out = ""
            for user in line[0]:
                out += "\'" + str(user) + "\', "

            out = out[:-2]
            print(out, file=outfile)


if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '12g') \
        .set('spark.executor.memory', '12g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw4')
    parser.add_argument('--filter_threshold', type=int, default=7)
    parser.add_argument('--input_file', type=str, default='ub_sample_data.csv')
    parser.add_argument('--betweenness_output_file', type=str, default='task2out1.txt')
    parser.add_argument('--community_output_file', type=str, default='task2out2.txt')

    args = parser.parse_args()

    start = time.time()
    main(args.filter_threshold, args.input_file, args.betweenness_output_file, args.community_output_file, sc)
    sc.stop()
    end = time.time()
    print("Runtime:", end-start)