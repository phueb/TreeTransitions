import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from treetransitions.params import DefaultParams, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


# fig
PLOT_NUM_SVS = 64
LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (8, 8)
DPI = None

# in-out correlation matrix
BINARY = False

# toy data
TRUNCATE_SIZE = 1
NUM_CATS = 32

params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_seqs = 5 * 10 ** 6
params.num_cats_list = [NUM_CATS]
params.truncate_num_cats = NUM_CATS
params.truncate_list = [0.5, 1.0]  # [1.0, 1.0] is okay
params.truncate_control = False
params.num_partitions = 8


toy_data = ToyData(params)


def plot_comparison(ys):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('truncate_list={}\ntruncate_control={}\ntruncate_num_cats={}'.format(
        params.truncate_list, params.truncate_control, params.truncate_num_cats), fontsize=AX_FONTSIZE)
    ax.set_ylabel('singular value', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    for n, y in enumerate(ys):
        ax.plot(y, label='partition {}'.format(n + 1), linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


def make_bigram_count_mat(id_seqs_mat):
    assert id_seqs_mat.shape[1] == 2  # works with bi-grams only
    print('Making In-Out matrix with shape={}...'.format(id_seqs_mat.shape))
    res = np.zeros((toy_data.num_vocab, toy_data.num_vocab))
    for row_id, col_id in zip(id_seqs_mat[:, 0], id_seqs_mat[:, 1]):
        res[row_id, col_id] += 1
    return res


# print results
singular_vals = []
for id_seq_mat_chunk in np.vsplit(toy_data.id_sequences_mat, params.num_partitions):
    in_out_corr_mat = make_bigram_count_mat(id_seq_mat_chunk)
    mat = normalize(in_out_corr_mat, axis=1, norm='l2', copy=False)
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
    singular_vals.append(pca.singular_values_[:PLOT_NUM_SVS])

# plot
plot_comparison(singular_vals)

print('\ntruncate_control={}'.format(params.truncate_control))



