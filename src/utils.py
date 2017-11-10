import logging
from pprint import pprint
import numpy as np
from collections import OrderedDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_top_k_aspect_clusters(aspect_clusters, X_tag_freq, k):
    cluster_sum_freq = []
    for cluster in aspect_clusters:
        cluster_mem_freq = [X_tag_freq[word]['freq'] for word in cluster]
        sum_freq = sum(cluster_mem_freq)
        cluster_sum_freq.append(sum_freq)
    sumfreq_to_aspcluster_map = zip(cluster_sum_freq, aspect_clusters)
    k_top_sumfreq_to_aspcluster_map = sorted(sumfreq_to_aspcluster_map, reverse=True, key=lambda x: x[0])[:k]
    k_top_aspect_clusters = [tup[1] for tup in k_top_sumfreq_to_aspcluster_map]
    aspect_dict = {cluster[0]: cluster for cluster in k_top_aspect_clusters}
    return aspect_dict


def load_gold_standard(filename):
    lines = [l.strip() for l in open(filename).readlines()]
    aspects = [l.split('\t')[0] for l in lines]
    features = [l.split('\t')[1][1:-1] for l in lines]
    features = [fstring.split(', ') for fstring in features]
    logger.info('{} aspects and {} features in gold standard'.format(len(aspects), len(features)))
    return aspects, features


def scoring(aspects):

    gold_standard = '../data/gold_standard/aspectCluster_Cell-phones.txt'
    gold_standard_aspects, gold_standard_features = load_gold_standard(gold_standard)

    aspect_N_gold = len(gold_standard_aspects)
    print 'aspect_N_gold', aspect_N_gold
    aspect_N_result = len(aspects)
    print 'aspect_N_result', aspect_N_result
    aspect_N_agree = sum([1 for element in aspects if element in gold_standard_aspects])
    print 'aspect_N_agree', aspect_N_agree
    aspect_precision = float(aspect_N_agree)/aspect_N_result
    print 'precision', aspect_precision
    aspect_recall = float(aspect_N_agree)/aspect_N_gold
    print 'recall', aspect_recall
    aspect_F_score = float(2*aspect_precision*aspect_recall)/(aspect_precision+aspect_recall)
    print 'F score', aspect_F_score



def get_cluster_label(X, features_clusters):
    y_label = OrderedDict()
    for w in X: y_label[w]=-1
    for ind, features in enumerate(features_clusters):
        print features
        for w in X:
            print w
            if w in features:
                print '{} in features_cluster, labeled {}'.format(w, ind)
                y_label[w] = ind
                raw_input()
    return y_label.values()