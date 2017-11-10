from requests import get
from math import isinf
from joblib import Parallel,delayed
import numpy as np
import json
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"
def get_swoogle_semantic_sim (i,j,
                              s1, s2,
                              type='relation', corpus='webbase'):
    if i > j:
        return 0
    if i == j:
        return 1
    try:
        response = get(sss_url,
                       params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
        logger.debug('{} {} {}'.format(s1, s2, float(response.text.strip())))
        if isinf(float(response.text.strip())):
            return 0.0
        else:
            return float(response.text.strip())
    except:
        logger.error('Error in getting similarity for %s and %s' % (s1, s2))
        return 0.0


def get_semantic_sim_mat(words):
    G = []
    for i,wi in enumerate(words):
        #row = Parallel(n_jobs=20)(delayed(get_swoogle_semantic_sim)(i,j,wi,wj)
        #                           for j, wj in enumerate(words))
        row = []
        for j, wj in enumerate(words):
            swoogle_sim = get_swoogle_semantic_sim(i,j,wi,wj)
            row.append(swoogle_sim)
        G.append(row)
    G = np.array(G)
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            if i > j:
                G[i,j] = G[j,i]
    return G


if __name__ == '__main__':
    pass