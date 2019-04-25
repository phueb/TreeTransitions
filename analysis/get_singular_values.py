import numpy as np
from sklearn.decomposition import PCA
from scipy import sparse
from cytoolz import itertoolz
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from treetransitions.jobs import generate_toy_data
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


# fig
LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (10, 4)
DPI = None

# in-out correlation matrix
NGRAM_SIZE_REPRESENTATION = 1
BINARY = False

# toy data
TRUNCATE_SIZE = 1
NUM_CATS = 32
SHUFFLE_TOKENS = False  # TODO careful here

params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_tokens = 1 * 10 ** 6
params.num_levels = 10
params.e = 0.2
params.num_cats_list = [NUM_CATS]
params.truncate_num_cats = NUM_CATS
params.truncate_list = [0.5, 1.0]  # [1.0, 1.0] is okay
params.truncate_control = False
params.num_partitions = 2


toy_data = generate_toy_data(params)

if SHUFFLE_TOKENS:
    print('WARNING: Shuffling tokens')
    np.random.shuffle(toy_data.tokens)


def plot_comparison(ys):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('truncate_list={}'.format(
        params.truncate_list), fontsize=AX_FONTSIZE)
    ax.set_ylabel('singular value', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component #', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    for n, y in enumerate(ys):
        ax.plot(y, label='partition {}'.format(n + 1), linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


def make_in_out_corr_mat(tokens_part):
    print('Making in_out_corr_mat with partition sized={}'.format(len(tokens_part)))
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
    return res


# print results
singular_vals = []
part_size = params.num_tokens // params.num_partitions
for part in itertoolz.partition_all(part_size, toy_data.tokens):
    if len(part) != part_size:
            continue
    in_out_corr_mat = make_in_out_corr_mat(part)
    mat = normalize(in_out_corr_mat, axis=1, norm='l2', copy=False).todense()
    #
    pca = PCA(svd_solver='full')  # pca is invariant to transposition
    pca.fit(mat)
    # console
    print('total var={:,}'.format(np.var(mat, ddof=1, axis=0).sum().round(0)))  # total variance
    for start, end in [(0, 31), (31, 64), (64, 1023)]:
        print('start={} end={}'.format(start, end))

        print('var={:,} %var={:.2f} sgl-val={:,}'.format(
            np.sum(pca.explained_variance_[start:end]).round(0),
            np.sum(pca.explained_variance_ratio_[start:end]).round(2),
            np.sum(pca.singular_values_[start:end]).round(0)))
    print()
    singular_vals.append(pca.singular_values_)

# plot
plot_comparison(singular_vals)

print('\ntruncate_control={}'.format(params.truncate_control))
print('\nSHUFFLING_TOKENS={}'.format(SHUFFLE_TOKENS))



