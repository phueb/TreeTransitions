import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


# fig
PLOT_NUM_SVS = 64
LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (8, 8)
DPI = None


CLIPPING = True

Params.learning_rate = [0.04]
Params.mutation_prob = [0.01]
Params.num_seqs = [1 * 10 ** 6]
Params.truncate_list = [[0.75, 1.0]]
Params.num_partitions = [4]
Params.truncate_num_cats = [64]  # TODO
Params.truncate_control = ['none']  # TODO
Params.truncate_sign = [1]


def plot_comparison(ys, params):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('truncate_list={}\ntruncate_control={}\n'
              'truncate_num_cats={}\ntruncate_type={}\nmutation_prob={}'.format(
        params.truncate_list, params.truncate_control,
        params.truncate_num_cats, params.truncate_type, params.mutation_prob),
        fontsize=AX_FONTSIZE)
    ax.set_ylabel('Singular value', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.set_ylim([0, 14])
    # plot
    for n, y in enumerate(ys):
        ax.plot(y, label='partition {}'.format(n + 1), linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


def make_bigram_count_mat(word_seqs_mat, x_words, y_words):
    assert word_seqs_mat.shape[1] == 2  # works with bi-grams only
    print('Making In-Out matrix with shape={}...'.format(word_seqs_mat.shape))
    num_xws = len(x_words)
    num_yws = len(y_words)
    res = np.zeros((num_xws, num_yws))
    for xw, yw in word_seqs_mat:
        res[x_words.index(xw), y_words.index(yw)] += 1
    return res


def plot_heatmap(mat, ytick_labels, xtick_labels,
                 figsize=(10, 10), dpi=None, ticklabel_fs=1, title_fs=5):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.title('', fontsize=title_fs)
    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('jet'),
              interpolation='nearest')
    # xticks
    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=ticklabel_fs)
    # yticks
    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(ytick_labels,   # no need to reverse (because no extent is set)
                            rotation=0, fontsize=ticklabel_fs)
    # remove ticklines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()


for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
    params = ObjectView(param2val)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)
    toy_data = ToyData(params, max_ba=False, make_tokens=True)

    # # use legals mat as input to PCA
    # singular_vals = []
    # mat1 = toy_data.untruncated_legals_mat
    # pca1 = PCA(svd_solver='full')  # pca is invariant to transposition
    # pca1.fit(mat1)
    # singular_vals.append(pca1.singular_values_[:PLOT_NUM_SVS])
    # # plot
    # plot_comparison(singular_vals, params)

    # use in_out correlation mat computed on tokens as input to PCA
    singular_vals = []
    for word_seq_mat_chunk in np.vsplit(toy_data.word_sequences_mat, params.num_partitions):
        in_out_corr_mat = make_bigram_count_mat(word_seq_mat_chunk, toy_data.x_words, toy_data.y_words)
        if CLIPPING:
            mat2 = np.clip(in_out_corr_mat, 0, 1)
        else:
            mat2 = in_out_corr_mat

        # num_one_in_mat1 = len(np.where(mat1 == 1)[0])
        # num_one_in_mat2 = np.count_nonzero(mat2)
        # print('num 1s in legals_mat={:,}'.format(num_one_in_mat1))
        # print('num 1s in in_out_corr_mat={:,}'.format(num_one_in_mat2))
        # print('difference={:,}'.format(num_one_in_mat1 - num_one_in_mat2))

        pca2 = PCA(svd_solver='full')  # pca is invariant to transposition
        pca2.fit(mat2)

        # console
        print('total var={:,}'.format(np.var(mat2, ddof=1, axis=0).sum().round(0)))  # total variance
        for start, end in [(0, 31), (31, 64), (64, 1023)]:
            print('start={} end={}'.format(start, end))

            print('var={:,} %var={:.2f} sgl-val={:,}'.format(
                np.sum(pca2.explained_variance_[start:end]).round(0),
                np.sum(pca2.explained_variance_ratio_[start:end]).round(2),
                np.sum(pca2.singular_values_[start:end]).round(0)))
        print()
        singular_vals.append(pca2.singular_values_[:PLOT_NUM_SVS])
    # plot
    plot_comparison(singular_vals, params)