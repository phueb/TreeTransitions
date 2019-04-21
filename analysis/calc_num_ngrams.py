from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from cytoolz import itertoolz

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_vocab, make_legal_mats
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


TRUNCATE_SIZE = 1
NUM_CATS = 32


params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_tokens = 1 * 10 ** 5
params.num_levels = 10
params.e = 0.2
params.truncate_num_cats = NUM_CATS

params.truncate_list = [1.0, 0.9]

vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

# make underlying hierarchical structure
size2word2legals, ngram2legals_mat = make_legal_mats(
    vocab, params.num_descendants, params.num_levels, params.mutation_prob, params.max_ngram_size)

# probes_data
num_cats2word2sorted_legals = {}
print('Getting {} categories with parent_count={}...'.format(NUM_CATS, params.parent_count))
legals_mat = ngram2legals_mat[params.structure_ngram_size]
probes, probe2cat, word2sorted_legals = make_probe_data(
    vocab, size2word2legals[TRUNCATE_SIZE], legals_mat, NUM_CATS, params.parent_count, plot=False)
print('Collected {} probes'.format(len(probes)))
# check probe sim
probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
print('input-data row-wise ba={:.3f}'.format(ba1))
print('input-data col-wise ba={:.3f}'.format(ba2))
print()
num_cats2word2sorted_legals[NUM_CATS] = word2sorted_legals


# sample tokens
tokens = make_tokens(vocab, size2word2legals, num_cats2word2sorted_legals[params.truncate_num_cats],
                     params.num_tokens, params.max_ngram_size, params.truncate_list)

# n-grams
ngrams = list(itertoolz.sliding_window(params.max_ngram_size + 1, tokens))
num_ngrams = len(set(ngrams))
print('num_ngrams={:,}'.format(num_ngrams))