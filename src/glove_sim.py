from joblib import Parallel,delayed
import numpy as np
from pprint import pprint
import logging
from gensim.models.keyedvectors import KeyedVectors
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

t0 = time()

#word_vectors = KeyedVectors.load_word2vec_format('../../../../media/CFC6-79FC/data/GoogleNews-vectors-negative300.bin.gz',
#                                                 binary=True, limit=500000)
word_vectors = KeyedVectors.load_word2vec_format('../../../../media/CFC6-79FC/data/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec',
                                                 binary=False, limit=500000)
#word_vectors = KeyedVectors.load_word2vec_format('../data/w2v_format_glove.6B.50d.txt', binary=False)

logger.info('loaded word2vec vectors in {} sec.'.format(round(time()-t0,2)))
def get_glove_sim (i,j,s1,s2):
    if i > j:
        return 0
    if i == j:
        return 1
    try:
        return word_vectors.similarity(s1,s2)
    except:
        logger.error('Error in getting similarity for %s and %s' % (s1, s2))
        return 0.0


def get_semantic_sim_mat(words):
    G = []
    for i,wi in enumerate(words):
        # row = Parallel(n_jobs=20)(delayed(get_glove_sim)(i,j,wi,wj)
        #                           for j, wj in enumerate(words))

        row = []
        for j, wj in enumerate(words):
            row.append(get_glove_sim(i,j,wi,wj))
        G.append(row)
    G = np.array(G)
    pprint (G)
    raw_input()
    G[G < 0] = 0
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            if i > j:
                G[i,j] = G[j,i]
    return G


if __name__ == '__main__':
    pass