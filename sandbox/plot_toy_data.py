from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from treetransitions.figs import plot_heatmap
from treetransitions.utils import to_corr_mat, cluster
from treetransitions.toy_data import ToyData
from treetransitions.params import param2default
from treetransitions.job import Params

COMPLETE_LEGAL_MAT_LEGAL_PROB = 0.5

PLOT_TREES = True
PLOT_LEGALS_MAT = False
PLOT__COMPLETE_LEGALS_MAT = True
PLOT_CORR_MATS = True
PLOT_PROBES_CORR_MAT_WITH_LABELS = False
PLOT_PROBES_CORR_MAT_DG = True

params = Params.from_param2val(param2default)
toy_data = ToyData(params, max_ba=False, make_sequences=False)

if PLOT_TREES:
    for num_cats in toy_data.params.num_cats_list:
        toy_data.plot_tree(num_cats)


if PLOT_LEGALS_MAT:
    for name, legals_mat in toy_data.name2legals_mat.items():
        xlabel = '{}-words'.format(name)
        plot_heatmap(legals_mat, [], [], xlabel=xlabel)

if PLOT__COMPLETE_LEGALS_MAT:
    xlabel = 'Sequence-initial words'
    title = r'$P_1={}$'.format(COMPLETE_LEGAL_MAT_LEGAL_PROB)
    complete_legals_mat = toy_data.make_complete_legals_mat(legal_prob=COMPLETE_LEGAL_MAT_LEGAL_PROB)
    plot_heatmap(complete_legals_mat, [], [], xlabel=xlabel, title=title)


if PLOT_CORR_MATS:
    for name, legals_mat in toy_data.name2legals_mat.items():
        # correlation matrix is symmetric (xlabel=ylabel)
        # the only way to show hierarchical pattern here is to correlate context words
        # this is true because context words have vectors with hierarchical structure, not x-words
        xlabel = 'Context words'.format(name)
        plot_heatmap(cluster(to_corr_mat(legals_mat)), [], [], xlabel=xlabel)


if PLOT_PROBES_CORR_MAT_WITH_LABELS:
    corr_mat = to_corr_mat(toy_data.probes_legals_mat)
    clustered_corr_mat, row_words, col_words = cluster(corr_mat, toy_data.vocab, toy_data.vocab)
    plot_heatmap(clustered_corr_mat, row_words, col_words)  # row_words, col_words

# plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
if PLOT_PROBES_CORR_MAT_DG:
    corr_mat = to_corr_mat(toy_data.probes_legals_mat)
    z = linkage(corr_mat, metric='correlation')
    fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
    dendrogram(z, ax=ax)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.xlabel('word ids in vocab')
    plt.ylabel('distance')
    plt.show()
