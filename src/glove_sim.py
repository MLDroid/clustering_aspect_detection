from joblib import Parallel,delayed
import numpy as np
from pprint import pprint
import logging
from gensim.models.keyedvectors import KeyedVectors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


word_vectors = KeyedVectors.load_word2vec_format('../data/w2v_format_glove.6B.50d.txt',
                                                 binary=False)

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
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        row = Parallel(n_jobs=20)(delayed(get_glove_sim)(i,j,wi,wj)
                                  for j, wj in enumerate(words))

        # row = []
        # for j, wj in enumerate(words):
        #     row.append(get_glove_sim(i,j,wi,wj))
        G.append(row)
    G = np.array(G)
    # pprint (G)
    # raw_input()
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            if i > j:
                G[i,j] = G[j,i]
    return G


if __name__ == '__main__':
    pass