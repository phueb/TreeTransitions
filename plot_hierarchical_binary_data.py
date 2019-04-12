import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from treetransitions.hierarchical_data_utils import sample_from_hierarchical_diffusion, to_corr_mat, plot_heatmap, cluster

num_descendants = 2  # 2
num_levels = 12  # 10
mutation_prob = 0.2  # 0.05, the higher, the more unique rows in data (and lower first PC)

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

# plot_heatmap(mat, [], [])
plot_heatmap(clustered_corr_mat, row_words, col_words)

# plot dg - of the CORRELATION MATRIX  - NOT THE RAW DATA MATRIX
z = linkage(corr_mat, metric='correlation')  # TODO hierarchical clustering of corr_mat - is this cleaner?
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