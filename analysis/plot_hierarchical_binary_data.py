import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

from treetransitions.params import Params, ObjectView
from treetransitions.utils import to_corr_mat
from treetransitions.toy_data import ToyData
from treetransitions import config

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[1.0, 1.0]]


def cluster(m, original_row_words=None, original_col_words=None):
    print('Clustering...')
    #
    lnk0 = linkage(pdist(m))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)
    z = m[dg0['leaves'], :]  # reorder rows
    #
    lnk1 = linkage(pdist(m.T))
    dg1 = dendrogram(lnk1,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    z = z[:, dg1['leaves']]  # reorder cols
    #
    if original_row_words is None and original_col_words is None:
        return z
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return z, row_labels, col_labels


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


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)
    toy_data = ToyData(params, max_ba=False if config.Eval.debug else True)

    # corr_mat
    corr_mat = to_corr_mat(toy_data.legals_mat)

    clustered_corr_mat, row_words, col_words = cluster(corr_mat, toy_data.vocab, toy_data.vocab)
    plot_heatmap(clustered_corr_mat, row_words, col_words)

    # plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
    z = linkage(corr_mat, metric='correlation')
    fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
    dendrogram(z, ax=ax)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.xlabel('word ids in vocab')
    plt.ylabel('distance')
    plt.show()