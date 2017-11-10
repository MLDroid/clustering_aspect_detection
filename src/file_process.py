from collections import defaultdict
from nltk.corpus import stopwords
from pprint import pprint
from time import time
import re,logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stopwords = set(stopwords.words('english'))

def get_parsed_reviews(fname):
    parsed_reviews = []
    for line in open(fname):
        if line.startswith('parsed_review: '):
            review = line.split('parsed_review: ')[1].strip()[1:-1].split(', ')
            review = (tuple(w_pos.split('/')) for w_pos in review)
            processed_rev = []
            for tup in review:
                w = tup[0].lower()
                pos_tag = tup[1]
                if w not in stopwords:
                    processed_rev.append((w,pos_tag))
            parsed_reviews.append(processed_rev)
    return parsed_reviews

def get_X_tag_freq(parsed_reviews):
    t1 = time()
    X_tag_freq = defaultdict(dict)
    pos_to_choose = {'NN','NNP','JJ'}
    review_texts = []
    for rev in parsed_reviews:
        for word,pos_tag in rev:
            if pos_tag in pos_to_choose:
                X_tag_freq[word]['POS'] = pos_tag
                review_texts.append(word)
    for word in X_tag_freq:
        X_tag_freq[word]['freq'] = review_texts.count(word)
    X_tag_freq = {k:v for k,v in X_tag_freq.iteritems() if re.sub(r'\W+', '', k)}
    logger.info('get X_tag_freq dictionary in {} sec.'.format(round(time() - t1, 2)))
    return X_tag_freq

def get_enron_X_freq(parsed_email_bodys, X):
    t1 = time()
    X_freq = defaultdict(dict)
    review_texts = []
    for email in parsed_email_bodys:
        for word,pos_tag in email:
            if word in X:
                review_texts.append(word)
    for word in X:
        X_freq[word]['freq'] = review_texts.count(word)
    X_freq = {k:v for k,v in X_freq.iteritems() if re.sub(r'\W+', '', k)}
    logger.info('get X_freq dictionary in {} sec.'.format(round(time() - t1, 2)))
    return X_freq

def get_most_common_terms (X_tag_freq, s):
    freq_w_tuples = [(v['freq'],k) for k,v in X_tag_freq.iteritems()]
    freq_w_tuples.sort(reverse=True)
    top_n_freq_w_tuples = freq_w_tuples[:s]
    logger.debug('top frequent terms: {}'.format(top_n_freq_w_tuples))
    return [tup[1] for tup in top_n_freq_w_tuples]




if __name__ == '__main__':
    pass