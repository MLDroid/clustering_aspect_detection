import os,sys,json,re
import logging
from file_process import *
from pprint import pprint

logging.basicConfig(level=logging.INFO)
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
    cand_feats = list(set(X_tag_freq.keys()) - set(cand_aspects))
    X = cand_aspects + cand_feats
    print cand_aspects
    print cand_feats
    print X

if __name__ == '__main__':
    main ()