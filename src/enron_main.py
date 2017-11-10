import os,sys,json,re
import logging
import numpy as np
from file_process import *
from pprint import pprint
from semantic_sim import get_semantic_sim_mat
#from glove_sim import get_semantic_sim_mat
# from umbc_pretrained_vec_sim import get_semantic_sim_mat
from statistical_sim import get_statistical_sim_mat
from combine_G_T import get_combined_sim_dict
from clustering import cluster_aspects, cluster_features
from utils import get_top_k_aspect_clusters
from enron_dataset_process import get_parsed_email_bodys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parsed_email_bodys = []

def main ():
    w_t = 0.2
    w_gt = 0.5
    w_g = 1 - w_t - w_gt

    s = 200
    k = s
    delta = 0.33 #0.8  #0.33

    Email_dir = '../../data/Allfiles/'
    extn = '.txt'
    parsed_email_bodys = get_parsed_email_bodys(Email_dir,extn)

    feats_fname = '../../data/Top100_Allaspect_unigram.txt'
    X = [x.replace('\r\n','') for x in open(feats_fname, 'r').readlines()]
    logger.info('identified {} terms from {}: {}'.format(len(X),feats_fname,X))

    X_freq = get_enron_X_freq(parsed_email_bodys,X)
    X = sorted([(X_freq[w]['freq'],w) for w in X],
               reverse=True)
    X = [tup[1] for tup in X]

    cand_aspects = X[:s]
    logger.info('identified {} aspects terms: {}'.format(len(cand_aspects), cand_aspects))
    cand_feats = X[s:]
    logger.info('identified {} feature terms: {}'.format(len(cand_feats), cand_feats))

    # G = get_semantic_sim_mat (X)
    # G.dump('G_enron_swoogle_matrix.dat')
    # T = get_statistical_sim_mat (X, parsed_email_bodys)
    # T.dump('T_enron_matrix.dat')

    G = np.load('G_enron_swoogle_matrix.dat')
    pprint(G)
    T = np.load('T_enron_matrix.dat')
    pprint(T)


    comb_sim_dict = get_combined_sim_dict (w_g,w_t,G,T,X)
    initial_seed_clusters = [[asp_word] for asp_word in cand_aspects]
    aspect_clusters = cluster_aspects (initial_seed_clusters, comb_sim_dict,
                                       X_freq, delta)

    aspect_dict = get_top_k_aspect_clusters(aspect_clusters, X_freq, k)
    pprint(aspect_dict)
    raw_input()

    aspect_feature_clusters = cluster_features(aspect_dict.values(), cand_feats,
                                              comb_sim_dict, X_freq, delta)
    pprint(aspect_feature_clusters)





if __name__ == '__main__':
    main ()