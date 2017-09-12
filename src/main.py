import os,sys,json,re
import logging
from file_process import *
from pprint import pprint
from semantic_sim import get_semantic_sim_mat
from statistical_sim import get_statistical_sim_mat
from combine_G_T import get_combined_sim_dict
from distance import get_distance
from utils import get_k_aspect_to_features_dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parsed_reviews = []


def main ():
    s = 10
    k = 5
    delta = 0.3

    dataset_fname = '../data/reviews_Cell-phones.txt'
    parsed_reviews = get_parsed_reviews(dataset_fname)
    X_tag_freq = get_X_tag_freq (parsed_reviews)
    cand_aspects = get_most_common_terms(X_tag_freq, s)
    logger.info('identified {} aspect terms: {}'.format(len(cand_aspects),cand_aspects))

    cand_feats = list(set(X_tag_freq.keys()) - set(cand_aspects))
    logger.info('identified {} feature terms: {}'.format(len(cand_feats), cand_feats))
    X = cand_aspects + cand_feats

    X = X[:s] #just to make quick inferences during development
    G = get_semantic_sim_mat (X)
    pprint(G)
    T = get_statistical_sim_mat (X, parsed_reviews)
    comb_sim_dict = get_combined_sim_dict (G,T,X)

    Clusters = [[asp] for asp in cand_aspects]
    mergeable_pair_cluster = 'flag'
    while mergeable_pair_cluster:
        for ind,cluster_i in enumerate(Clusters):
            cluster_i_dist_map = {}
            for cluster_j in Clusters[ind+1:]:
                cluster_dist = get_distance(cluster_i,cluster_j,comb_sim_dict,X_tag_freq)
                cluster_i_dist_map[cluster_dist]= (cluster_i, cluster_j)
            if cluster_i_dist_map and min(cluster_i_dist_map.keys()) < delta:
                mergeable_pair_cluster = cluster_i_dist_map[min(cluster_i_dist_map.keys())]
                mergeable_pair_cluster[1].extend(cluster_i)
                Clusters.remove(cluster_i)
            else:
                mergeable_pair_cluster = None

    pprint(Clusters)

    #k_aspect_to_features_dict = get_k_aspect_to_features_dict(Clusters, X_tag_freq, k)
    #pprint(k_aspect_to_features_dict)




if __name__ == '__main__':
    main ()