import os,sys,json,re
import logging
import numpy as np
from file_process import *
from pprint import pprint
from combine_G_T import get_combined_sim_dict
from clustering import cluster_aspects, cluster_features
from utils import get_top_k_aspect_clusters, load_gold_standard, get_cluster_label
from enron_dataset_process import get_parsed_email_bodys
from sklearn.metrics import adjusted_rand_score

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV



class MyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, w_g=0, w_t=0, delta=0):
        """
        Called when initializing the classifier
        """
        self.w_g = w_g
        self.w_t = w_t
        self.delta = delta

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """

        comb_sim_dict = get_combined_sim_dict(self.w_g,self.w_t,G, T, X)
        initial_seed_clusters = [[asp_word] for asp_word in X[:s]]
        print initial_seed_clusters
        pprint(X_tag_freq)
        raw_input()
        aspect_clusters = cluster_aspects(initial_seed_clusters, comb_sim_dict,
                                          X_tag_freq, self.delta)

        aspect_dict = get_top_k_aspect_clusters(aspect_clusters, X_tag_freq, k)

        # cand_feats = X - aspect_dict.values()
        cand_feats = X[s:]
        aspect_feature_clusters = cluster_features(aspect_dict.values(), cand_feats,comb_sim_dict, X_tag_freq, self.delta)

        self.features_ = aspect_feature_clusters
        pprint(self.features_)
        self.aspects_ = aspect_dict.keys()
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "aspects_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        predict_label = get_cluster_label(X, self.features_)
        print predict_label
        raw_input()
        return(predict_label)

    def score(self, X, y=None):
        my_score = adjusted_rand_score(self.predict(X), y)
        print my_score
        raw_input()
        return my_score


s = 500
k = 46

def main():

    gold_standard = '../data/gold_standard/aspectCluster_Cell-phones.txt'
    gold_standard_aspects, gold_standard_features = load_gold_standard(gold_standard)
    num_of_features = sum([1 for features in gold_standard_features for feat in features ])
    logger.info('{} features in gold standard'.format(num_of_features))

    dataset_fname = '../data/reviews_Cell-phones.txt'
    parsed_reviews = get_parsed_reviews(dataset_fname)
    global X_tag_freq
    X_tag_freq = get_X_tag_freq(parsed_reviews)


    X = sorted([(X_tag_freq[w]['freq'],w) for w in X_tag_freq.keys()],
               reverse=True)
    X = [tup[1] for tup in X]

    y_label = get_cluster_label(X, gold_standard_features)

    global G,T
    G = np.load('G_matrix.dat')
    pprint(G.shape)
    T = np.load('T_matrix.dat')
    pprint(T.shape)

    # tuned_params = {'w_g':[0.1,0.2,0.3,0.4,0.5,0.6],
    #                 'w_t':[0.1,0.2,0.3,0.4,0.5,0.6],
    #                 'delta':[0.01, 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}

    tuned_params = {'w_g':[0.3, 0.4],
                    'w_t':[0.4],
                    'delta':[0.3]}

    gs = GridSearchCV(MyClassifier(), tuned_params)
    gs.fit(X, y_label)

    print gs.best_params_



if __name__ == '__main__':
    main ()