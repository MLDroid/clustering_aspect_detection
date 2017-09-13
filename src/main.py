import os,sys,json,re
import logging
from file_process import *
from pprint import pprint
# from semantic_sim import get_semantic_sim_mat
from glove_sim import get_semantic_sim_mat
from statistical_sim import get_statistical_sim_mat
from combine_G_T import get_combined_sim_dict
from utils import get_k_aspect_to_features_dict
from clustering import cluster_aspects


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parsed_reviews = []

def main ():
    s = 10
    k = 5
    delta = 0.08

    dataset_fname = '../data/reviews_Cell-phones.txt'
    parsed_reviews = get_parsed_reviews(dataset_fname)
    X_tag_freq = get_X_tag_freq (parsed_reviews)
    cand_aspects = get_most_common_terms(X_tag_freq, s)
    logger.info('identified {} aspect terms: {}'.format(len(cand_aspects),cand_aspects))

    cand_feats = list(set(X_tag_freq.keys()) - set(cand_aspects))
    logger.info('identified {} feature terms: {}'.format(len(cand_feats), cand_feats))
    X = sorted([(X_tag_freq[w]['freq'],w) for w in X_tag_freq],
               reverse=True)
    X = [tup[1] for tup in X]
    # print X
    # raw_input()

    X = X[:s+30] #just to make quick inferences during development
    G = get_semantic_sim_mat (X)
    T = get_statistical_sim_mat (X, parsed_reviews)
    comb_sim_dict = get_combined_sim_dict (G,T,X)
    initial_seed_clusters = [[asp_word] for asp_word in X[:s]]
    aspect_clusters = cluster_aspects (initial_seed_clusters, comb_sim_dict,
                                       X_tag_freq, delta)
    initial_aspect_feature_clusters = aspect_clusters + [[feat] for feat in X[s:]]
    aspect_feature_clusters = cluster_aspects(initial_aspect_feature_clusters,
                                              comb_sim_dict,X_tag_freq, delta)

    return aspect_feature_clusters


    #k_aspect_to_features_dict = get_k_aspect_to_features_dict(Clusters, X_tag_freq, k)
    #pprint(k_aspect_to_features_dict)




if __name__ == '__main__':
    main ()