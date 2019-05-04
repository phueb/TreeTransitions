import numpy as np
import pyprind
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import multiprocessing as mp
from cytoolz import itertoolz
from matplotlib.colors import to_hex
from collections import Counter
import random
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.utils import to_corr_mat, calc_ba
from treetransitions import config


def make_sequences_chunk(x_words, y_words, chunk_id, xw_2yws, xw2sorted_yws, num_seqs, truncate, truncate_type):
    if chunk_id % config.Eval.num_processes == 0:
        print('\nMaking sequences chunk with truncate={} and truncate_type={}'.format(truncate, truncate_type))
    #
    assert truncate_type in ['legals', 'probes']
    seq_size = 2
    res = np.random.choice(x_words, size=(num_seqs, seq_size))
    pbar = pyprind.ProgBar(num_seqs) if chunk_id % config.Eval.num_processes == 0 else None
    num_y_word_insertions = 0
    for seq in res:
        if truncate_type == 'probes':
            if np.random.binomial(n=1, p=1 - truncate, size=1).item():
                seq[:] = np.random.choice(y_words, size=2)
                num_y_word_insertions += 1
                continue
            else:
                xw = seq[0]
                yw = np.random.choice(xw_2yws[xw], size=1, p=None).item()
                seq[-1] = yw
        elif truncate_type == 'legals':
            # get words which are legal to come next - y_words
            xw = seq[0]
            sorted_yws = xw2sorted_yws[xw]
            num_truncated = int(len(sorted_yws) * truncate)
            legal_yws = set(sorted_yws[:num_truncated])
            legal_yws.intersection_update(xw_2yws[xw])
            # sample from legal y_words
            yw = np.random.choice(list(legal_yws), size=1, p=None).item()
            seq[-1] = yw
        else:
            raise AttributeError('Invalid arg to "truncate_type".')
        pbar.update() if chunk_id % config.Eval.num_processes == 0 else None
    if chunk_id % config.Eval.num_processes == 0:
        print('num_seqs={} num_y_word_insertions={} prob={}'.format(
            num_seqs, num_y_word_insertions, num_y_word_insertions / num_seqs))
    return res


class ToyData:
    def __init__(self, params, max_ba=True):
        self.params = params
        self.x_words = self.make_vocab(1)  # probes
        self.y_words = self.make_vocab(0)  # non-probes (or context words)
        self.num_yws = len(self.y_words)
        self.num_xws = len(self.x_words)
        #
        self.vocab = list(set(self.y_words + self.x_words))
        self.num_vocab = len(self.vocab)
        self.word2id = {word: n for n, word in enumerate(self.vocab)}
        #
        self.yw2node0 = self.make_yw2node0()
        self.legals_mat = self.make_legals_mat()
        self.xw2yws = self.make_xw2yws()
        #
        self.z = self.make_z()
        self.probes = self.make_probes()
        self.num_cats2probe2cat = {num_cats: self.make_probe2cat(num_cats)
                                   for num_cats in params.num_cats_list}
        #
        self.num_cats2cat2yws = {num_cats: self.make_cat2yws(num_cats)
                                 for num_cats in params.num_cats_list}
        self.num_cats2xw2_sorted_yws = {num_cats: self.make_xw2_sorted_yws()
                                        for num_cats in params.num_cats_list}
        self.num_cats2max_ba = self.make_num_cats2max_ba() if max_ba else None
        #
        self.num_cats2cmap = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1)
                              for num_cats in params.num_cats_list}
        self.num_cats2probe2color = {num_cats: self.make_probe2color(num_cats)
                                     for num_cats in params.num_cats_list}
        for num_cats in self.params.num_cats_list:
            self.plot_tree(num_cats) if config.Eval.plot_tree else None
        #
        self.word_sequences_mat = self.make_sequences_mat()
        self.num_seqs = len(self.word_sequences_mat)  # divisible by mb_size and num_partitions
        print('shape of word_sequences_mat={}'.format(self.word_sequences_mat.shape))
        self.id_sequences_mat = np.asarray([[self.word2id[w] for w in seq]
                                            for seq in self.word_sequences_mat]).reshape((self.num_seqs, -1))

    def make_vocab(self, vocab_id):
        num_vocab = self.params.num_descendants ** self.params.num_levels
        vocab = ['w{}'.format(i) for i in np.arange(num_vocab * vocab_id, num_vocab * (vocab_id + 1))]
        return vocab

    def make_yw2node0(self):
        res = {}
        for yw in self.y_words:
            node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
            res[yw] = node0
        return res

    def make_legals_mat(self):
        """
        each row contains a vector sampled from hierarchical process.
        each column represents which y_words are allowed to follow an x_word (word associated with column)
        """
        res = np.zeros((self.num_yws, self.num_xws), dtype=np.int)
        for n, yw in enumerate(self.y_words):
            res[n, :] = self.sample_from_hierarchical_diffusion(self.yw2node0[yw])
        print('Done')
        return res

    def make_xw2yws(self):
        res = {}
        for xw, vals in zip(self.x_words, self.legals_mat.T):
            yws = [yw for yw, val in zip(self.y_words, vals) if val == self.yw2node0[yw]]  # node0 is required
            res[xw] = yws
        return res

    # ///////////////////////////////////////////////////////////////////////// probes

    def make_z(self, method='single', metric='euclidean'):
        # careful: idx1 and idx2 in res are not integers (they are floats)
        corr_mat = to_corr_mat(self.legals_mat)
        res = linkage(corr_mat, metric=metric, method=method)  # need to cluster correlation matrix otherwise messy
        return res

    def get_all_probes_in_tree(self, res1, z, node_id):
        """
        # z should be the result of linkage,which returns an array of length n - 1
        # giving you information about the n - 1 cluster merges which it needs to pairwise merge n clusters.
        # Z[i] will tell us which clusters were merged in the i-th iteration.
        # a row in z is: [idx1, idx2, distance, count]
        # all idx >= len(X) actually refer to the cluster formed in Z[idx - len(X)]
        """
        try:
            p = self.x_words[node_id.astype(int)]
        except IndexError:  # in case idx does not refer to leaf node (then it refers to cluster)
            new_node_id1 = z[node_id.astype(int) - self.num_xws][0]
            new_node_id2 = z[node_id.astype(int) - self.num_xws][1]
            self.get_all_probes_in_tree(res1, z, new_node_id1)
            self.get_all_probes_in_tree(res1, z, new_node_id2)
        else:
            res1.append(p)

    def make_probes(self):
        # find cluster (identified by idx1 and idx2) with parent_count nodes beneath it
        for row in self.z:
            idx1, idx2, dist, count = row
            if count == self.params.parent_count:
                break
        else:
            raise RuntimeError('Did not find any cluster with count={}'.format(self.params.parent_count))
        # get probes - tree structure is preserved in order of how probes are retrieved from tree
        res = []
        for idx in [idx1, idx2]:
            self.get_all_probes_in_tree(res, self.z, idx)
        print('Collected {} probes'.format(len(res)))
        assert set(self.x_words) == set(res)
        return res

    def make_probe2cat(self, num_cats):
        print('Assigning probes to {} categories'.format(num_cats))
        num_members = self.params.parent_count / num_cats
        assert num_members.is_integer()
        num_members = int(num_members)
        #
        assert num_cats % 2 == 0
        res = {}

        for cat, cat_probes in enumerate(itertoolz.partition_all(num_members, self.probes)):
            assert len(cat_probes) == num_members
            res.update({p: cat for p in cat_probes})
        return res

    def make_probe2color(self, num_cats):
        res = {}
        cmap = self.num_cats2cmap[num_cats]
        for p in self.probes:
            cat = self.num_cats2probe2cat[num_cats][p]
            res[p] = to_hex(cmap(cat))
        return res

    def make_cat2yws(self, num_cats):
        res = {cat: [] for cat in range(num_cats)}
        probe2cat = self.num_cats2probe2cat[num_cats]
        for p, cat in probe2cat.items():
            yws = self.xw2yws[p]
            res[cat] += yws
        return res

    # /////////////////////////////////////////////////////////////// sequences

    def make_xw2_sorted_yws(self):
        assert isinstance(self.params.truncate_control, bool)  # if [False], it would incorrectly evaluate to True
        print('truncate_control={}'.format(self.params.truncate_control))
        #
        res = {}
        probe2cat = self.num_cats2probe2cat[self.params.truncate_num_cats]
        cat2yws = self.num_cats2cat2yws[self.params.truncate_num_cats]
        for xw in self.x_words:
            cat = probe2cat[xw]
            cat_yws = cat2yws[cat]
            cat_yws2freq = Counter(cat_yws)
            #
            sorted_yws = sorted(set(cat_yws), key=cat_yws2freq.get)  # sorts in ascending order
            if self.params.truncate_control:
                random.shuffle(sorted_yws)
            res[xw] = sorted_yws
        return res

    def sample_from_hierarchical_diffusion(self, node0):
        """the higher the change probability (e),
         the less variance accounted for by higher-up nodes"""
        nodes = [node0]
        for level in range(self.params.num_levels):
            candidate_nodes = nodes * self.params.num_descendants
            s = len(candidate_nodes)
            nodes = [node if p else -node
                     for node, p in zip(candidate_nodes,
                                        np.random.binomial(n=2, p=1 - self.params.mutation_prob, size=s))]
        return nodes

    def make_sequences_mat(self, num_chunks=32):
        """
        a sequence is like a document - no statistical regularities span across document boundaries
        each word is constrained by the legals matrices - which are hierarchical -
        and determine the legal successors for each word in the vocabulary.
        there is one legal matrix for each ngram (in other words, for each distance up to max_ngram_size)
        the set of legal words that can follow a word is the intersection of legal words dictated by each legals matrix.
        the size of the intersection is approximately the same for each word, and a word is sampled uniformly from this set
        the size of the intersection is the best possible perplexity a model can achieve,
        because perplexity indicates the number of choices from a random uniformly distributed set of choices
        """
        num_processes = config.Eval.num_processes
        # make sequences - in parallel
        pool = mp.Pool(processes=num_processes)
        min_truncate, max_truncate = self.params.truncate_list
        # truncate_steps = np.linspace(min_truncate, max_truncate, num_chunks + 2)[1:-1]
        truncate_steps = np.linspace(min_truncate, max_truncate, num_chunks)
        num_seqs_in_chunk = self.params.num_seqs // num_chunks
        xw2sorted_yws = self.num_cats2xw2_sorted_yws[self.params.truncate_num_cats]
        results = [pool.apply_async(
            make_sequences_chunk,
            args=(self.x_words, self.y_words, chunk_id, self.xw2yws, xw2sorted_yws,
                  num_seqs_in_chunk, truncate_steps[chunk_id], self.params.truncate_type))
            for chunk_id in range(num_chunks)]
        chunks = []
        print('Creating sequences...')
        try:
            for n, r in enumerate(results):
                chunk = r.get()
                if n % num_processes == 0:
                    print('\nnum types in chunk={}'.format(len(set(np.hstack(chunk)))))
                chunks.append(chunk)
            pool.close()
        except KeyboardInterrupt:
            pool.close()
            raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
        print('Done')
        # stack
        stacked = np.vstack(chunks)
        # make divisible
        num_remainder = len(stacked) % (self.params.mb_size * self.params.num_partitions)
        num_divisible = len(stacked) - num_remainder
        print('Shortened num_seqs to {:,}'.format(num_divisible))
        assert len(stacked) % self.params.mb_size == 0.0
        return stacked[:num_divisible]

    def gen_part_id_seqs(self):
        for part_seq in np.vsplit(self.id_sequences_mat, self.params.num_partitions):
            print('Shape of part_id_seq={}'.format(part_seq.shape))
            yield part_seq

    # //////////////////////////////////////////////////////////// misc

    def make_num_cats2max_ba(self):
        res = {}
        for num_cats in self.params.num_cats_list:
            probes = self.probes
            probe2cat = self.num_cats2probe2cat[num_cats]
            # probe_acts1 = self.legals_mat[[self.word2id1[p] for p in probes], :]
            # ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
            probe_acts2 = self.legals_mat[:, [self.x_words.index(p) for p in probes]].T
            ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
            # print('input-data row-wise ba={:.3f}'.format(ba1))
            print('input-data col-wise ba={:.3f}'.format(ba2))
            print()
            res[num_cats] = ba2
        return res

    def plot_tree(self, num_cats):
        probe2color = self.num_cats2probe2color[num_cats]
        link2color = {}
        for i, i12 in enumerate(self.z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(self.z)
                      else probe2color[self.vocab[x.astype(int)]]
                      for x in i12)
            link2color[i + 1 + len(self.z)] = c1 if c1 == c2 else 'grey'
        #
        cmap = self.num_cats2cmap[num_cats]
        fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
        dg = dendrogram(self.z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        reordered_vocab = np.asarray(self.vocab)[dg['leaves']]
        ax.set_xticklabels([w if w in self.probes else '' for w in reordered_vocab], fontsize=6)
        # assign x tick label color
        probe2_cat = self.num_cats2probe2cat[num_cats]
        colors = [to_hex(cmap(i)) for i in range(num_cats)]
        for n, w in enumerate(self.vocab):
            try:
                cat = probe2_cat[w]
            except KeyError:
                continue
            else:
                color = colors[cat]
                ax.get_xticklabels()[n].set_color(color)  # TODO doesn't work
        plt.show()
        #
        # clustered_corr_mat = corr_mat[dg['leaves'], :]
        # clustered_corr_mat = clustered_corr_mat[:, dg['leaves']]
        # plot_heatmap(clustered_corr_mat, [], [], dpi=None)
        # plot_heatmap(cluster(data_mat), [], [], dpi=None)
