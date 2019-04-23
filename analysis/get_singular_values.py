import numpy as np
from sklearn.decomposition import PCA
from scipy import sparse
from cytoolz import itertoolz

from treetransitions.jobs import generate_toy_data
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
params.truncate_list = [0.5, 1.0]
params.truncate_control = [False]  # TODO careful here


toy_data = generate_toy_data(params, NUM_CATS)


# ///////////////////////////////////////////////////////////////////


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens_part = toy_data.tokens[start:end]
    # types
    types = sorted(set(toy_data.tokens))
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

# print results
for mat in [in_out_corr_mat1.todense(),
            in_out_corr_mat2.todense()]:  # pca is invariant to transposition
    pca = PCA(svd_solver='full')
    pca.fit(mat)
    # console
    print('total var={:,}'.format(np.var(mat, ddof=1, axis=0).sum().round(0)))  # total variance
    for start, end in [(0, 31), (31, 64), (64, 1023)]:
    # for start in np.arange(0, 1024, 32):
    #     end = start + 32
        print('start={} end={}'.format(start, end))

        print('var={:,} %var={:.2f} sgl-val={:,}'.format(
            np.sum(pca.explained_variance_[start:end]).round(0),
            np.sum(pca.explained_variance_ratio_[start:end]).round(2),
            np.sum(pca.singular_values_[start:end]).round(0)))
    print()

print('\ntruncate_control={}'.format(params.truncate_control))



