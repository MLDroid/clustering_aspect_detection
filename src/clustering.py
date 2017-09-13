import os,sys,json,re
import logging
from file_process import *
from pprint import pprint
from distance import get_distance
import operator
from copy import deepcopy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cluster_aspects(existing_clusters, sim_dict, word_tagfreq_dict, delta):
    phi = deepcopy(existing_clusters)
    processed_cluster_flag = False
    while not processed_cluster_flag:
        clusters_to_loop_thro = deepcopy(phi)
        for i,cluster_i in enumerate(clusters_to_loop_thro):
            cluster_i_dist_map = {}
            for j,cluster_j in enumerate(clusters_to_loop_thro[i+1:]):
                cluster_dist = get_distance(cluster_i,cluster_j,
                                            sim_dict,word_tagfreq_dict)
                cluster_i_dist_map[(i, i+1+j)] = cluster_dist
            if cluster_i_dist_map:
                index_tup_of_closest_clusters, dict_bw_closest_cluster = \
                    sorted(cluster_i_dist_map.items(),key=operator.itemgetter(1))[0]
                del cluster_i_dist_map
                if dict_bw_closest_cluster < delta:
                    phi[index_tup_of_closest_clusters[0]].\
                        extend(phi[index_tup_of_closest_clusters[1]])
                    del phi[index_tup_of_closest_clusters[1]]
                    break
                else:
                    continue
            else:
                processed_cluster_flag = True
                break
    pprint (phi)
    return phi


'''
def cluster_aspects (asp_words, sim_dict, word_tagfreq_dict, delta):
    clusters = {asp:[asp] for asp in asp_words}
    new_clusters = []
    while True:
        for i,cluster_center_i in enumerate(clusters.iterkeys()):
            cluster_i = clusters[cluster_center_i]
            cluster_i_dist_map = {}
            for cluster_center_j in clusters.keys()[i+1:]:
                cluster_j = clusters[cluster_center_j]
                cluster_dist = get_distance(cluster_i,cluster_j,
                                            sim_dict,word_tagfreq_dict)
                cluster_i_dist_map[(cluster_center_i,cluster_center_j)] = cluster_dist
            if cluster_i_dist_map:
                cluster_i_dist_map = sorted(cluster_i_dist_map.items(),
                                            key=operator.itemgetter(1))
                min_dist_bw_clusters = cluster_i_dist_map.values()[0]
                potentially_mergable_cluster = cluster_i_dist_map.keys()[0]
                del cluster_i_dist_map
                if min_dist_bw_clusters < delta:
                    mergeable_pair_cluster = potentially_mergable_cluster

                else:
                    mergeable_pair_cluster = None

'''

