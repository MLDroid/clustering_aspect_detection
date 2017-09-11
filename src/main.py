import os,sys,json,re
import logging
from file_process import *
from pprint import pprint
from semantic_sim import get_semantic_sim_mat
from statistical_sim import get_statistical_sim_mat

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
parsed_reviews = []


def main ():
    s = 10
    k = 20
    delta = 0.8

    dataset_fname = '../data/reviews_Cell-phones.txt'
    parsed_reviews = get_parsed_reviews(dataset_fname)
    X_tag_freq = get_X_tag_freq (parsed_reviews)
    cand_aspects = get_most_common_terms(X_tag_freq, s)
    logger.info('identified {} aspect terms: {}'.format(len(cand_aspects),cand_aspects))

    cand_feats = list(set(X_tag_freq.keys()) - set(cand_aspects))
    logger.info('identified {} feature terms: {}'.format(len(cand_feats), cand_feats))
    X = cand_aspects + cand_feats

    X = X[:10] #just to make quick inferences during development
    # G = get_semantic_sim_mat (X)
    # pprint(G)
    T = get_statistical_sim_mat (X, parsed_reviews, X_tag_freq)
    # sim_mat = get_sim_mat (G,T)


if __name__ == '__main__':
    main ()