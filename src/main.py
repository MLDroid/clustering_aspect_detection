import os,sys,json,re
import logging
import numpy as np
from file_process import *
from pprint import pprint
from semantic_sim import get_semantic_sim_mat
#from glove_sim import get_semantic_sim_mat
#from umbc_pretrained_vec_sim import get_semantic_sim_mat
from statistical_sim import get_statistical_sim_mat
from combine_G_T import get_combined_sim_dict
from clustering import cluster_aspects, cluster_features
from utils import get_top_k_aspect_clusters, scoring
from enron_dataset_process import get_parsed_email_bodys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parsed_reviews = []

def main ():
    w_t = 0.2
    w_gt = 0.5
    w_g = 1 - w_t - w_gt

    s = 500
    k = 50
    delta = 0.8  #0.5

    dataset_fname = '../data/reviews_Cell-phones.txt'
    parsed_reviews = get_parsed_reviews(dataset_fname)
    X_tag_freq = get_X_tag_freq(parsed_reviews)

    cand_aspects = get_most_common_terms(X_tag_freq, s)
    logger.info('identified {} aspect terms: {}'.format(len(cand_aspects), cand_aspects))


    cand_feats = list(set(X_tag_freq.keys()) - set(cand_aspects))
    logger.info('identified {} feature terms: {}'.format(len(cand_feats), cand_feats))
    X = sorted([(X_tag_freq[w]['freq'],w) for w in X_tag_freq],
               reverse=True)
    X = [tup[1] for tup in X]

    #X = X[:s+1] #just to make quick inferences during development
    # G = get_semantic_sim_mat (X)
    # G.dump('G_all_corpus_swoogle_matrix.dat')
    # T = get_statistical_sim_mat (X, parsed_reviews)
    # T.dump('T_enron_matrix.dat')

    G = np.load('G_matrix.dat')
    pprint(G.shape)
    T = np.load('T_matrix.dat')
    pprint(T.shape)
    raw_input()

    comb_sim_dict = get_combined_sim_dict (w_g,w_t,G,T,X)
    initial_seed_clusters = [[asp_word] for asp_word in X[:s]]
    aspect_clusters = cluster_aspects (initial_seed_clusters, comb_sim_dict,
                                       X_tag_freq, delta)

    aspect_dict = get_top_k_aspect_clusters(aspect_clusters, X_tag_freq, k)
    pprint(aspect_dict)
    # raw_input()

    scoring(aspect_dict.keys())
    #
    # aspect_feature_clusters = cluster_features(aspect_dict.values(), X[s:],
    #                                           comb_sim_dict, X_tag_freq, delta)
    # pprint(aspect_feature_clusters)







if __name__ == '__main__':
    main ()