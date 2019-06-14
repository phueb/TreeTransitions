import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


# fig
PLOT_NUM_SVS = 64
LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (5, 5)
DPI = None


Params.mutation_prob = [0.01]
Params.num_seqs = [1 * 10 ** 6]
Params.num_partitions = [2]
Params.legal_probs = [[0.5, 1.0]]
Params.num_non_probes_list = [[1024], [1024, 1024], [1024, 1024, 1024]]
Params.num_contexts = [1024]


def plot_comparison(ys, params):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    # plt.title('num_non_probes_list={}'.format(params.num_non_probes_list))
    plt.title('number of syntactic categories={}'.format(len(params.num_non_probes_list)))
    ax.set_ylabel('Singular value', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Singular Dimension', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.set_ylim([0, 14])
    x = np.arange(PLOT_NUM_SVS)
    # ax.set_xticks(x)
    # ax.set_xticklabels(x)
    # plot
    labels = iter([r'$P_1={}$'.format(sp) for sp in params.legal_probs])
    for n, y in enumerate(ys):
        ax.plot(x, y, label=next(labels) or 'partition {}'.format(n + 1), linewidth=2)
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


for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
    params = ObjectView(param2val)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)
    toy_data = ToyData(params, max_ba=False, make_tokens=True)

    # use in_out correlation mat computed on tokens as input to PCA
    singular_vals = []
    for word_seq_mat_chunk in np.vsplit(toy_data.word_sequences_mat, params.num_partitions):

        all_x_words = []
        all_y_words = []
        for name, (xws, yws) in toy_data.name2words.items():
            all_x_words.extend(xws)
            all_y_words.extend(yws)
        term_by_window_mat = make_bigram_count_mat(word_seq_mat_chunk, all_x_words, all_y_words)

        term_by_window_mat = normalize(term_by_window_mat, axis=1, norm='l2', copy=False)
        pca2 = PCA(svd_solver='full')  # pca is invariant to transposition
        pca2.fit(term_by_window_mat)

        # console
        print('total var={:,}'.format(np.var(term_by_window_mat, ddof=1, axis=0).sum().round(0)))  # total variance
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