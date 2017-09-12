from requests import get
from math import isinf
from joblib import Parallel,delayed
import numpy as np
from pprint import pprint
import logging
from collections import Counter
from math import log10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def break_review_into_sents(parsed_reviews):
#     reviews_as_sents = []
#     pos_tags_to_consider = {'NN','NNP','JJ'}
#     for rev in parsed_reviews:
#         sents = [[] for _ in xrange(rev.count(('eos','eos')))]
#         sent_index = 0
#         for word,pos in rev:
#             if pos == 'eos':
#                 sent_index += 1
#             elif pos in pos_tags_to_consider:
#                 sents[sent_index].append(word)
#             else:
#                 continue
#         reviews_as_sents.append(sents)
#     return reviews_as_sents

def break_review_into_sents(parsed_reviews):
    reviews_as_sents = {doc_id:[] for doc_id in xrange(len(parsed_reviews))}
    pos_tags_to_consider = {'NN','NNP','JJ'}
    for doc_id, rev_doc in enumerate(parsed_reviews):
        doc_as_sents_list = [[] for _ in xrange(rev_doc.count(('eos','eos')))]
        sent_id = 0
        for word,pos in rev_doc:
            if pos == 'eos':
                sent_id += 1
            elif pos in pos_tags_to_consider:
                doc_as_sents_list[sent_id].append(word)
            else:
                continue
        reviews_as_sents[doc_id] = doc_as_sents_list
    return reviews_as_sents


def get_sentence_level_bigram_counter (parsed_reviews):
    reviews_as_sents = break_review_into_sents(parsed_reviews)
    bigrams = [set() for doc_id in reviews_as_sents]
    for doc_id, rev_doc in reviews_as_sents.iteritems():
        for sent in rev_doc:
            for i, w_i in enumerate(sent):
                for j, w_j in enumerate(sent[i + 1:]):
                    bigrams[doc_id].add((w_i, w_j))
                    bigrams[doc_id].add((w_j, w_i))
    bigrams = [bg for bigram_set in bigrams for bg in bigram_set]
    bigram_doc_counter = Counter(bigrams)
    return bigram_doc_counter

def get_word_docfreq_count(words, parsed_reviews):
    word_docfreq_count = {w:0 for w in words}
    words_to_count_docfreq = set(words)
    for rev in parsed_reviews:
        all_words_in_rev = {w for w,pos in rev}
        for w in all_words_in_rev:
            if w not in words_to_count_docfreq:
                continue
            word_docfreq_count[w] += 1
    return word_docfreq_count


def get_statistical_sim_mat (words, parsed_reviews):
    f_xi = get_word_docfreq_count(words, parsed_reviews)
    f_xi_xj = get_sentence_level_bigram_counter(parsed_reviews)

    N = float(len(parsed_reviews)) # num of docs
    T = np.zeros(shape=(len(words),len(words)))
    for i, wi in enumerate(words):
        for j, wj in enumerate(words):
            if i == j:
                T[i, j] = 1.0
                continue
            if i < j:
                fxixj = f_xi_xj.get((wi, wj),0)
                if 0 == fxixj:
                    T[i, j] = 0
                    continue
                f_wi = f_xi[wi]
                f_wj = f_xi[wj]
                nr = log10( (N * fxixj) / (f_wi * f_wj) )
                dr = -log10(fxixj / N)
                npmi = nr / dr
                T[i, j] = npmi
            else:
                T[i, j] = T[j, i]

    T = (T + 1) / 2.0
    # pprint(T)
    # raw_input()
    return T



if __name__ == '__main__':
    pass