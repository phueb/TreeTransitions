import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from cytoolz import itertoolz
from matplotlib.colors import to_hex
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.params import Params


class ToyData:
    def __init__(self,
                 params: Params,
                 ):
        self.params = params

        self.num_x = params.num_x_words
        self.num_y = params.num_x_words

        self.xws = self.make_words('x', self.num_x)
        self.yws = self.make_words('y', self.num_y)

        self.vocab = self.xws + self.yws
        self.num_vocab = len(self.vocab)  # used by rnn
        self.word2id = {word: n for n, word in enumerate(self.vocab)}

        # the probability of co-occurrence of x and y words
        # this is where the hierarchical structure of the data is created
        ul = lr = np.random.random((self.num_x // 2, self.num_y // 2)) * 1.0
        ll = ur = np.random.random((self.num_x // 2, self.num_y // 2)) * 0.5
        self.p_mat = np.block([
            [ul, ur],
            [ll, lr],
        ])

        # linkage
        self.z = self.make_z()
        self.num_cats2probe2cat = {num_cats: self.make_probe2cat(num_cats)
                                   for num_cats in params.num_cats_list}

        # sequences
        self.id_sequences_mat = None  # TODO

        # plot
        self.num_cats2cmap = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1) for num_cats in params.num_cats_list}
        self.num_cats2probe2color = {num_cats: self.make_probe2color(num_cats) for num_cats in params.num_cats_list}

    @staticmethod
    def make_words(prefix: str,
                   num: int,
                   ):
        vocab = ['{}{}'.format(prefix, i) for i in np.arange(num)]
        return vocab

    def make_z(self,
               method='single',
               metric='euclidean',
               ):
        # careful: idx1 and idx2 in res are not integers (they are floats)
        res = linkage(cosine_similarity(self.p_mat), metric=metric, method=method)  # TODO cluster the p_mat?
        return res

    def get_all_probes_in_tree(self, res1, z, node_id):
        """
        # z should be the result of linkage,which returns an array of length n - 1
        # giving you information about the n - 1 cluster merges which it needs to pairwise merge n clusters.
        # Z[i] will tell us which clusters were merged in the i-th iteration.
        # a row in z is: [idx1, idx2, distance, count]
        # all idx >= len(X) actually refer to the cluster formed in Z[idx - len(X)]
        """
        num_xws = len(self.xws)
        try:
            p = self.xws[node_id.astype(int)]
        except IndexError:  # in case idx does not refer to leaf node (then it refers to cluster)
            new_node_id1 = z[node_id.astype(int) - num_xws][0]
            new_node_id2 = z[node_id.astype(int) - num_xws][1]
            self.get_all_probes_in_tree(res1, z, new_node_id1)
            self.get_all_probes_in_tree(res1, z, new_node_id2)
        else:
            res1.append(p)

    def make_probe2cat(self, num_cats):

        # find cluster (identified by idx1 and idx2) with num_x_words nodes beneath it
        for row in self.z:
            idx1, idx2, dist, count = row
            if count == self.params.num_x_words:
                break
        else:
            raise RuntimeError('Did not find any cluster with count={}'.format(self.params.num_x_words))

        # get xwords in correct order - tree structure is preserved in order of how probes are retrieved from tree
        ordered_xws = []
        for idx in [idx1, idx2]:
            self.get_all_probes_in_tree(ordered_xws, self.z, idx)
        print('Collected {} probes'.format(len(ordered_xws)))
        assert set(self.xws) == set(ordered_xws)

        #
        print('Assigning probes to {} categories'.format(num_cats))
        num_members = self.params.num_x_words / num_cats
        assert num_members.is_integer()
        num_members = int(num_members)

        #
        assert num_cats % 2 == 0
        res = {}
        for cat, cat_probes in enumerate(itertoolz.partition_all(num_members, ordered_xws)):
            assert len(cat_probes) == num_members
            res.update({p: cat for p in cat_probes})

        return res

    def make_probe2color(self, num_cats):
        res = {}
        cmap = self.num_cats2cmap[num_cats]
        for p in self.xws:
            cat = self.num_cats2probe2cat[num_cats][p]
            res[p] = to_hex(cmap(cat))
        return res

    def plot_tree(self, num_cats):
        probe2color = self.num_cats2probe2color[num_cats]
        link2color = {}
        for i, i12 in enumerate(self.z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(self.z)
                      else probe2color[self.xws[x.astype(int)]]
                      for x in i12)
            link2color[i + 1 + len(self.z)] = c1 if c1 == c2 else 'grey'
        # plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        dendrogram(self.z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        ax.set_xticklabels([])
        plt.show()
