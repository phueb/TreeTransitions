import numpy as np
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from cytoolz import itertoolz

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_vocab, make_legal_mats
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


# in-out correlation matrix
NGRAM_SIZE_REPRESENTATION = 1
BINARY = False

# toy data
TRUNCATE_SIZE = 1
NUM_CATS = 32  # TODO careful here
params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_tokens = 1 * 10 ** 6
params.num_levels = 10
params.e = 0.2
params.truncate_num_cats = NUM_CATS
params.truncate_list = [0.1, 1.0]
params.truncate_control = [False]  # TODO careful here
params.num_partitions = [8]


vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

# make underlying hierarchical structure
size2word2legals, ngram2legals_mat = make_legal_mats(
    vocab, params.num_descendants, params.num_levels, params.mutation_prob, params.max_ngram_size)

# probes_data
num_cats2word2sorted_legals = {}
print('Getting {} categories with parent_count={}...'.format(NUM_CATS, params.parent_count))
legals_mat = ngram2legals_mat[params.structure_ngram_size]
probes, probe2cat, word2sorted_legals = make_probe_data(
    vocab, size2word2legals[TRUNCATE_SIZE], legals_mat, NUM_CATS, params.parent_count, params.truncate_control)
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


# ///////////////////////////////////////////////////////////////////


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens_part = tokens[start:end]
    # types
    types = sorted(set(tokens))
    num_types = len(types)
    type2id = {t: n for n, t in enumerate(types)}
    # ngrams
    ngrams = list(itertoolz.sliding_window(NGRAM_SIZE_REPRESENTATION, tokens_part))
    ngram_types = sorted(set(ngrams))
    num_ngram_types = len(ngram_types)
    ngram2id = {t: n for n, t in enumerate(ngram_types)}
    # make sparse matrix (types in rows, ngrams in cols)
    shape = (num_types, num_ngram_types)
    print('Making In-Out matrix with shape={}...'.format(shape))
    data = []
    row_ids = []
    cold_ids = []
    mat_loc2freq = {}  # to keep track of number of ngram & type co-occurrence
    for n, ngram in enumerate(ngrams[:-NGRAM_SIZE_REPRESENTATION]):
        # row_id + col_id
        col_id = ngram2id[ngram]
        next_ngram = ngrams[n + 1]
        next_type = next_ngram[-1]
        row_id = type2id[next_type]
        # freq
        try:
            freq = mat_loc2freq[(row_id, col_id)]
        except KeyError:
            mat_loc2freq[(row_id, col_id)] = 1
            freq = 1
        else:
            mat_loc2freq[(row_id, col_id)] += 1
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(1 if BINARY else freq)
    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (row_ids, cold_ids)), shape=(num_types, num_ngram_types))
    return res, types


midpoint_loc = params.num_tokens // 2
start1, end1 = 0, midpoint_loc // 1
start2, end2 = params.num_tokens - end1, params.num_tokens

# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'tokens between\n{:,} & {:,}'.format(start2, end2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(start1, end1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(start2, end2)


def printout(fitted):
    print('(  :31) sum of sing. values={:,} | %var={:.2f}'.format(
        np.sum(fitted.singular_values_[:31]).round(0),
        np.sum(fitted.explained_variance_ratio_[:31])))
    print('(31:64) sum of sing. values={:,} | %var={:.2f}'.format(
        np.sum(fitted.singular_values_[31:64]).round(0),
        np.sum(fitted.explained_variance_ratio_[31:64])))
    print('(64:  ) sum of sing. values={:,} | %var={:.2f}'.format(
        np.sum(fitted.singular_values_[64:]).round(0),
        np.sum(fitted.explained_variance_ratio_[64:])))


# pca1
print('Fitting PCA 1 ...')
pca1 = PCA(n_components=None)
pca1.fit(in_out_corr_mat1.todense())
printout(pca1)

# pca2
print('Fitting PCA 2 ...')
pca2 = PCA(n_components=None)
pca2.fit_transform(in_out_corr_mat2.todense())
printout(pca2)

print('\ntruncate_control={}'.format(params.truncate_control))



