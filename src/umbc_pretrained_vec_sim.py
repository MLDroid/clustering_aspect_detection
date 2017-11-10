from joblib import Parallel,delayed
import numpy as np
from pprint import pprint
import logging
from gensim.models.keyedvectors import KeyedVectors
from time import time
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

umbc_fname = '../../../../media/CFC6-79FC/data/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec'

def get_umbc_sim (i,j,v1,v2):
    if i > j:
        return 0
    if i == j:
        return 1
    else:
        return float(cosine_similarity(v1,v2))



def get_semantic_sim_mat(words):
    t0 = time()
    umbc_lines_to_consider = [l.strip() for l in open(umbc_fname).readlines()
                              if l.split()[0] in words]
    umbc_w_vect_dict = {l.split()[0]: np.array([float(n) for n in l.split()[1:]])
                        for l in umbc_lines_to_consider}
    logger.info('loaded umbc vectors in {} sec.'.format(round(time() - t0, 2)))

    G = []
    for i,wi in enumerate(words):
        row = []
        for j, wj in enumerate(words):
            if wi in umbc_w_vect_dict.keys() and wj in umbc_w_vect_dict.keys():
                vi = umbc_w_vect_dict.get(wi)
                vj = umbc_w_vect_dict.get(wj)
                row.append(get_umbc_sim(i,j,vi,vj))
            else:
                row.append(-1)
        G.append(row)
    G = np.array(G)
    G = (G + 1)/2.0
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            if i > j:
                G[i,j] = G[j,i]
    return G


if __name__ == '__main__':
    pass