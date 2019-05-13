import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import cycle

from treetransitions.utils import to_corr_mat


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


NUM_CATS = 32

NUM_DESCENDANTS = 2
STOP_MUTATION_LEVEL = 12
BOTTOM_MUTATION_PROB = 0.00
TOP_MUTATION_PROB = 1.0  # set to 1.0 or 0.5

NUM_VOCAB = 1024


def make_overwrite_ids(cat_id):
    res = [cat_id]
    num_top_levels = np.log2(NUM_CATS).astype(np.int) - 1
    for _ in range(num_top_levels):
        prev_id = res[-1] // 2
        res.append(prev_id)
    return res[::-1]


def make_nodes_template(cat_id):
    res = np.array([-1])
    for overwrite_id in make_overwrite_ids(cat_id):
        # repeat + overwrite
        repeated = np.repeat(res, NUM_DESCENDANTS)
        repeated[overwrite_id] = 1
        # mutate
        res = repeated * [1 if binom else -1
                     for binom in np.random.binomial(n=1, p=1 - TOP_MUTATION_PROB, size=len(repeated))]
    return res


def make_legals_row(res, end_num):
    while True:  # keep branching until end_num nodes are created
        if len(res) >= end_num:
            return res
        #
        rep = np.repeat(res, NUM_DESCENDANTS)
        res = rep * [1 if binom else -1
                     for binom in np.random.binomial(n=1, p=1 - BOTTOM_MUTATION_PROB, size=len(rep))]


def make_legals_mat():
    res = np.zeros((NUM_VOCAB, NUM_VOCAB), dtype=np.int)
    row_id = 0
    for cat_id in range(NUM_CATS):
        nodes_template = make_nodes_template(cat_id)
        print('nodes template')
        print(nodes_template)
        print('length of nodes_template={}'.format(len(nodes_template)))  # should be 32
        # make legals row for each member in category
        num_members = NUM_VOCAB // NUM_CATS
        for _ in range(num_members):
            legals_row = make_legals_row(nodes_template, NUM_VOCAB)
            res[row_id, :] = legals_row
            print(legals_row)
            row_id += 1
    print(res)
    return res


# corr_mat
legals_mat = make_legals_mat()
corr_mat = to_corr_mat(legals_mat)

# plot corr_mat
clustered_corr_mat = cluster(corr_mat)
plot_heatmap(clustered_corr_mat, [], [])

# plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
# z = linkage(corr_mat, metric='correlation')
# fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
# dendrogram(z, ax=ax)
# plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
# plt.xlabel('word ids in vocab')
# plt.ylabel('distance')
# plt.show()
