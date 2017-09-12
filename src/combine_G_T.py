import logging
from pprint import pprint
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

w_t = 0.2
w_gt = 0.6
w_g = 1 - w_t - w_gt

def get_wordpair_similarity(index_i, index_j, G, T):
    sim_g = cosine_similarity(G[index_i], G[index_j])
    sim_t = cosine_similarity(T[index_i], T[index_j])
    sim_gt = max(sim_g, sim_t)
    sim = w_g*sim_g + w_t*sim_t + w_gt*sim_gt
    return float(sim)

def get_combined_similarity (G,T):
    comb_sim_mat = np.zeros(shape=T.shape)
    for i in xrange(comb_sim_mat.shape[0]):
        for j in xrange(comb_sim_mat.shape[1]):
            if i == j:
                comb_sim_mat[i, j] = 1
            elif i < j:
                comb_sim_mat[i, j] = get_wordpair_similarity(i, j, G, T)
            else:
                comb_sim_mat[i, j] = comb_sim_mat[j, i]
    return comb_sim_mat

def get_combined_sim_dict(G,T,words):
    comb_sim_mat = get_combined_similarity (G,T)
    comb_sim_dict = {}
    for i,wi in enumerate(words):
        for j,wj in enumerate(words):
            comb_sim_dict[(wi,wj)] = comb_sim_mat[i,j]
    return comb_sim_dict


if __name__ == '__main__':
    pass