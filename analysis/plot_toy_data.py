import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from treetransitions.params import Params, ObjectView
from treetransitions.utils import to_corr_mat, cluster
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


Params.num_seqs = [2 * 10 ** 6]
Params.mutation_prob = [0.01]
Params.legal_probs = [[1.0, 1.0]]
Params.num_contexts = [128]
Params.num_non_probes_list = [[1024]]

COMPLETE_LEGAL_MAT_LEGAL_PROB = 0.5

PLOT_TREES = False
PLOT_LEGALS_MAT = False
PLOT__COMPLETE_LEGALS_MAT = True
PLOT_LEGAL_CORR_MATS = True
PLOT_WITH_LABELS = False
PLOT_CORR_MAT_DG = False


def plot_heatmap(mat, ytick_labels, xtick_labels, title='', xlabel='', ticklabel_fs=1, fontsize=16):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=None)
    plt.title(title, fontsize=fontsize)
    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap='jet',
              interpolation='nearest')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Context words', fontsize=fontsize)
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
    toy_data = ToyData(params, max_ba=False, make_sequences=False)

    if PLOT_TREES:
        for num_cats in toy_data.params.num_cats_list:
            toy_data.plot_tree(num_cats)

    # plot
    if PLOT_LEGALS_MAT:
        for name, legals_mat in toy_data.name2legals_mat.items():
            xlabel = '{}-words'.format(name)
            plot_heatmap(legals_mat, [], [], xlabel=xlabel)
    # plot
    if PLOT__COMPLETE_LEGALS_MAT:
        xlabel = 'Sequence-initial words'
        title = r'$P_1={}$'.format(COMPLETE_LEGAL_MAT_LEGAL_PROB)
        complete_legals_mat = toy_data.make_complete_legals_mat(legal_prob=COMPLETE_LEGAL_MAT_LEGAL_PROB)
        plot_heatmap(complete_legals_mat, [], [], xlabel=xlabel, title=title)
    # plot
    if PLOT_LEGAL_CORR_MATS:
        for name, legals_mat in toy_data.name2legals_mat.items():
            # correlation matrix is symmetric (xlabel=ylabel)
            # the only way to show hierarchical pattern here is to correlate context words
            # this is true because context words have vectors with hierarchical structure, not x-words
            xlabel = 'Context words'.format(name)
            plot_heatmap(cluster(to_corr_mat(legals_mat)), [], [], xlabel=xlabel)

    # corr_mat
    if PLOT_WITH_LABELS:
        corr_mat = to_corr_mat(toy_data.probes_legals_mat)
        clustered_corr_mat, row_words, col_words = cluster(corr_mat, toy_data.vocab, toy_data.vocab)
        plot_heatmap(clustered_corr_mat, row_words, col_words)  # row_words, col_words

    # plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
    if PLOT_CORR_MAT_DG:
        corr_mat = to_corr_mat(toy_data.probes_legals_mat)
        z = linkage(corr_mat, metric='correlation')
        fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
        dendrogram(z, ax=ax)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
        plt.xlabel('word ids in vocab')
        plt.ylabel('distance')
        plt.show()

    print('------------------------------------------------------------')
