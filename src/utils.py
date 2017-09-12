import logging
from pprint import pprint
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_k_aspect_to_features_dict(Clusters, X_tag_freq, k):
    k_aspect_to_features_dict = {}
    for cluster in Clusters:
        cluster_mem_freq = [X_tag_freq[word]['freq'] for word in cluster]
        rep_asp = cluster[cluster_mem_freq.index(max(cluster_mem_freq))]
        sum_freq_cluster = sum(cluster_mem_freq)
        map = zip(sum_freq_cluster, rep_asp, cluster)
        k_top_map_by_sumfreq = sorted(map, reverse=True, key=lambda x: x[0])[:k]
        for tup in k_top_map_by_sumfreq:
            k_aspect_to_features_dict[tup[1]]=tup[2]
    return k_aspect_to_features_dict