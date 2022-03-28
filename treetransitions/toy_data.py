from typing import List
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from cytoolz import itertoolz
from matplotlib.colors import to_hex

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
        self.token2id = {word: n for n, word in enumerate(self.vocab)}

        # the probability of co-occurrence of x and y words
        # this is where the hierarchical structure of the data is created
        self.p_mat = self.make_p_mat()

        # linkage
        self.z = self.make_z()
        self.num_cats2probe2cat = {num_cats: self.make_probe2cat(num_cats)
                                   for num_cats in params.num_cats_list}

        # sequences
        self.id_sequences_mat = self.make_id_sequences_mat()  # [num sequences, 2]

        # assign color map to each category structure
        self.num_cats2color_map = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1)
                                   for num_cats in params.num_cats_list}

    def make_p_mat(self,
                   plot: bool = False,
                   ) -> np.array:
        """
        return a matrix of probabilities.
        the value at index i and j defines the probability that x-word i and y-word j co-occur in the data.
        """

        res = np.ones((self.num_x, self.num_y), dtype=float)

        s_val = 1.0  # the value to be subtracted

        for num_cats in self.params.num_cats_list:
            print(f'Generating category structure with num_cats={num_cats}')

            num_subtractions = num_cats // 2
            sub_square_size = self.num_x // num_subtractions

            s_val -= 0.1  # the value to be subtracted is reduced at each lower level in the hierarchy

            assert 0.0 <= s_val <= 1.0

            for cat_id in range(num_subtractions):

                # make s
                # note: s is a matrix with 4 quadrants, and is used to "carve out" the big matrix
                ul = lr = np.ones((sub_square_size // 2, sub_square_size // 2)) * 0.0
                ll = ur = np.ones((sub_square_size // 2, sub_square_size // 2)) * s_val
                s = np.block([
                    [ul, ur],
                    [ll, lr],
                ])

                # subtract
                start = sub_square_size * cat_id
                p_mat_sub_square = res[start: start + sub_square_size, start: start + sub_square_size]
                p_mat_sub_square -= s

                # make sure all values are valid probabilities
                assert 0.0 <= np.min(res) <= 1.0
                assert 0.0 <= np.max(res) <= 1.0

            if plot:
                from treetransitions.figs import plot_heatmap
                plot_heatmap(res, [], [])

        return res

    def make_id_sequences_mat(self) -> np.array:
        """
        return a matrix of shape [num sequences, 2] used for training,
         where first item = x-word token ID, and second item = y-word token ID
        """

        sequences = []

        for xw in np.random.choice(self.xws, size=self.params.num_seqs):

            # get y-word
            probabilities = self.p_mat[self.token2id[xw], :] / self.p_mat[self.token2id[xw], :].sum()
            assert len(probabilities) == len(self.yws)
            yw = np.random.choice(self.yws, p=probabilities)

            xw_token_id = self.token2id[xw]
            yw_token_id = self.token2id[yw]

            # collect
            sequences.append([xw_token_id, yw_token_id])

        res = np.vstack(sequences)

        assert len(res) % self.params.num_parts == 0
        assert len(res) % self.params.batch_size == 0

        return res

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
        res = linkage(self.p_mat, metric=metric, method=method)
        return res

    def get_all_xws_in_tree(self,
                            ordered_xws_: List[str],
                            z: linkage,
                            node_id: int,
                            ):
        """
        # z should be the result of linkage,which returns an array of length n - 1
        # giving you information about the n - 1 cluster merges which it needs to pairwise merge n clusters.
        # Z[i] will tell us which clusters were merged in the i-th iteration.
        # a row in z is: [idx1, idx2, distance, count]
        # all idx >= len(X) actually refer to the cluster formed in Z[idx - len(X)]
        """

        num_xws = len(self.xws)
        try:
            p = self.xws[node_id]
        except IndexError:  # in case idx does not refer to leaf node (then it refers to cluster)
            new_node_id1 = z[node_id - num_xws][0]
            new_node_id2 = z[node_id - num_xws][1]
            self.get_all_xws_in_tree(ordered_xws_, z, int(new_node_id1))
            self.get_all_xws_in_tree(ordered_xws_, z, int(new_node_id2))
        else:
            ordered_xws_.append(p)

    def make_probe2cat(self, num_cats):
        """
        use linkage of p_mat to assign category-membership to x-words

        Note:
            this is strictly speaking, not necessary, but is cool
        """

        # find cluster (identified by idx1 and idx2) with num_x_words nodes beneath it
        for idx1, idx2, dist, count in self.z:
            if count == self.params.num_x_words:
                break
        else:
            raise RuntimeError('Did not find any cluster with count={}'.format(self.params.num_x_words))

        # get x-words in correct order - tree structure is preserved in order of how probes are retrieved from tree
        ordered_xws = []
        for node_id in [idx1, idx2]:
            self.get_all_xws_in_tree(ordered_xws, self.z, int(node_id))
        print('Collected {} x-words'.format(len(ordered_xws)))
        assert set(self.xws) == set(ordered_xws)

        #
        print('Assigning probes to {} categories'.format(num_cats))
        num_members = self.params.num_x_words / num_cats
        assert num_members.is_integer()
        num_members = int(num_members)

        # partition x-words into categories
        assert num_cats % 2 == 0
        res = {}
        for cat, cat_probes in enumerate(itertoolz.partition_all(num_members, ordered_xws)):
            assert len(cat_probes) == num_members
            res.update({p: cat for p in cat_probes})

        return res

    def plot_tree(self,
                  num_cats: int,
                  ):
        """
        plot dendrogram with same-category probes shown in same color
        """

        # assign each probe a color
        probe2color = {}
        color_map = self.num_cats2color_map[num_cats]
        for p in self.xws:
            cat = self.num_cats2probe2cat[num_cats][p]
            probe2color[p] = to_hex(color_map(cat))

        # define the link_color_func
        link2color = {}
        for i, i12 in enumerate(self.z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(self.z) else probe2color[self.xws[x.astype(int)]]
                      for x in i12)
            if c1 == c2:
                link2color[i + 1 + len(self.z)] = c1
            else:
                link2color[i + 1 + len(self.z)] = 'grey'

        # plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        dendrogram(self.z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        ax.set_xticklabels([])
        plt.show()

    def gen_part_id_seqs(self):
        """
        return sequences of token ids for training/

        Note:
            a sequence is a window of two items, Xi and Yi
        """

        for res in np.vsplit(self.id_sequences_mat, self.params.num_parts):
            print(f'Shape of matrix containing sequences in partition={res.shape}')
            yield res
