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
                index_tup_of_closest_clusters, dist_bw_closest_cluster = \
                    sorted(cluster_i_dist_map.items(),key=operator.itemgetter(1))[0]
                del cluster_i_dist_map
                if dist_bw_closest_cluster < delta:
                    phi[index_tup_of_closest_clusters[0]].\
                        extend(phi[index_tup_of_closest_clusters[1]])
                    del phi[index_tup_of_closest_clusters[1]]
                    break
                else:
                    continue
            else:
                processed_cluster_flag = True
                break
    return phi



def cluster_features(existing_clusters, candi_features, sim_dict, word_tagfreq_dict, delta):
    phi = deepcopy(existing_clusters)
    for word in candi_features:
        clusters_to_loop_thro = deepcopy(phi)
        word_dist_map = {}
        for i,cluster_i in enumerate(clusters_to_loop_thro):
            cluster_dist = get_distance(cluster_i,[word],
                                        sim_dict,word_tagfreq_dict)
            word_dist_map[i] = cluster_dist
        if word_dist_map:
            index_of_closest_cluster, dist_of_closest_cluster = \
                sorted(word_dist_map.items(),key=operator.itemgetter(1))[0]
            del word_dist_map
            if dist_of_closest_cluster < delta:
                phi[index_of_closest_cluster].append(word)
                break
            else:
                continue
    return phi


