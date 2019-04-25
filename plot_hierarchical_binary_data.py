import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from treetransitions.utils import to_corr_mat

num_descendants = 2  # 2
num_levels = 12  # 10
mutation_prob = 0.2  # 0.05, the higher, the more unique rows in data (and lower first PC)


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


# vocab
num_vocab = num_descendants ** num_levels
print('num_vocab={}'.format(num_vocab))
vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
word2id = {word: n for n, word in enumerate(vocab)}


# make data_mat
mat = np.zeros((num_vocab, num_vocab))
for n in range(num_vocab):
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    mat[n] = sample_from_hierarchical_diffusion(node0, num_descendants, num_levels, mutation_prob)
print(mat)
# corr_mat
corr_mat = to_corr_mat(mat)

clustered_corr_mat, row_words, col_words = cluster(corr_mat, vocab, vocab)
plot_heatmap(clustered_corr_mat, row_words, col_words)

# plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
z = linkage(corr_mat, metric='correlation')
fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
dendrogram(z, ax=ax)
plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
plt.xlabel('word ids in vocab')
plt.ylabel('distance')
plt.show()

# pca
pca = PCA()
fitter = pca.fit_transform(mat)
print(['{:.4f}'.format(i) for i in pca.explained_variance_ratio_][:10])