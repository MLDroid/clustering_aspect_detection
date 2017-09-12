import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_avg_dist (cluster_1, cluster_2, sim_dict):
    sum_dist = 0
    for w1 in cluster_1:
        for w2 in cluster_2:
            sum_dist += 1 - sim_dict[(w1,w2)]
    avg_dist = sum_dist / float(len(cluster_1)*len(cluster_2))
    return avg_dist

def get_rep_dist (cluster_1, cluster_2,
                sim_dict,word_tagfreq_map):
    cluster_1_freqs = [word_tagfreq_map[w]['freq'] for w in cluster_1]
    cluster_1_rep = cluster_1[cluster_1_freqs.index(max(cluster_1_freqs))]
    cluster_2_freqs = [word_tagfreq_map[w]['freq'] for w in cluster_2]
    cluster_2_rep = cluster_2[cluster_2_freqs.index(max(cluster_2_freqs))]
    rep_dist = 1 - sim_dict[(cluster_1_rep, cluster_2_rep)]
    return rep_dist

def get_distance(cluster_1, cluster_2, sim_dict, word_tagfreq_map):
    avg_dist = get_avg_dist (cluster_1, cluster_2, sim_dict)
    rep_dist = get_rep_dist (cluster_1, cluster_2,
                             sim_dict,word_tagfreq_map)
    comb_dist = max(avg_dist, rep_dist)
    return comb_dist

if __name__ == '__main__':
    pass