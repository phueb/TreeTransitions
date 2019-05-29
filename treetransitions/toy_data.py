import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import multiprocessing as mp
from cytoolz import itertoolz
from matplotlib.colors import to_hex
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.utils import to_corr_mat, calc_ba, cluster
from treetransitions import config


def make_sequences_chunk(x_words, y_words, num_seqs, legals_mat):
    print('\nMaking {} sequences...'.format(num_seqs))
    # xw2yws
    xw2yws = {}
    for xw, col in zip(x_words, legals_mat.T):
        yws = [yw for yw, val in zip(y_words, col) if val == 1] # if statement is required
        xw2yws[xw] = yws
    #
    seq_size = 2
    res = np.random.choice(x_words, size=(num_seqs, seq_size))
    for seq in res:
        xw = seq[0]
        yw = np.random.choice(xw2yws[xw], size=1, p=None).item()
        seq[-1] = yw
    return res


"""
legals_mat: 
each row contains a vector sampled from hierarchical process.
each column (x-word) represents which y_words are allowed to follow x_word
"""


class ToyData:
    def __init__(self, params, max_ba=True, make_tokens=True):
        self.params = params
        self.x_words = self.make_x_words()  # probes
        self.y_words = self.make_y_words()  # non-probes (or context words)
        self.num_yws = len(self.y_words)
        self.num_xws = len(self.x_words)
        #
        self.vocab = list(set(self.y_words + self.x_words))
        self.num_vocab = len(self.vocab)  # used by rnn
        self.word2id = {word: n for n, word in enumerate(self.vocab)}
        #
        self.node0 = 1  # TODO test node is always 1
        self.untruncated_legals_mat = self.make_legals_mat()
        #
        self.z = self.make_z()
        self.probes = self.make_probes()
        self.num_cats2probe2cat = {num_cats: self.make_probe2cat(num_cats)
                                   for num_cats in params.num_cats_list}
        #
        self.truncate_linspace = np.linspace(*self.params.truncate_list, self.params.num_partitions)
        self.legals_mats = [self.truncate_legals_mat(truncate) for truncate in self.truncate_linspace]  # TODO test
        # ba
        self.num_cats2max_ba = self.make_num_cats2max_ba() if max_ba else None
        # plot
        self.num_cats2cmap = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1) for num_cats in params.num_cats_list}
        self.num_cats2probe2color = {num_cats: self.make_probe2color(num_cats) for num_cats in params.num_cats_list}
        for num_cats in self.params.num_cats_list:
            self.plot_tree(num_cats) if config.Eval.plot_tree else None
        # plot legals_mat
        if config.Eval.plot_legals_mat:
            for legal_mat in self.legals_mats:
                # self.plot_heatmap(cluster(legal_mat))
                self.plot_heatmap(legal_mat)
        # pot legals_mat correlation matrix
        if config.Eval.plot_corr_mat:
            for legal_mat in self.legals_mats:
                self.plot_heatmap(cluster(to_corr_mat(legal_mat)))
        #
        if make_tokens:
            self.word_sequences_mat = self.make_sequences_mat()
            self.num_seqs = len(self.word_sequences_mat)  # divisible by mb_size and num_partitions
            print('shape of word_sequences_mat={}'.format(self.word_sequences_mat.shape))
            self.id_sequences_mat = np.asarray([[self.word2id[w] for w in seq]
                                                for seq in self.word_sequences_mat]).reshape((self.num_seqs, -1))

    def make_y_words(self):
        vocab = ['y{}'.format(i) for i in np.arange(self.params.num_contexts)]
        return vocab

    def make_x_words(self):
        vocab = ['x{}'.format(i) for i in np.arange(self.params.num_probes)]
        return vocab

    # /////////////////////////////////////////////////////////////// legals

    def make_legals_mat(self):
        """
        each row contains a vector sampled from hierarchical process.
        each column represents which y_words are allowed to follow an x_word (word associated with column)
        """
        res = np.zeros((self.num_yws, self.num_xws), dtype=np.int)
        for row_id, yw in enumerate(self.y_words):
            res[row_id, :] = self.sample_from_hierarchical_diffusion(self.node0)
        print('Done')
        return res

    def truncate_legals_mat(self, truncate_prob):
        print('Truncating with truncate_prob={}'.format(truncate_prob))
        # cat2col_ids
        probe2cat = self.num_cats2probe2cat[self.params.truncate_num_cats]
        cat2probes = {cat: [] for cat in probe2cat.values()}
        cat2col_ids = {cat: [] for cat in probe2cat.values()}
        for p in self.probes:
            cat = probe2cat[p]
            cat2probes[cat].append(p)
            cat2col_ids[cat].append(self.x_words.index(p))
        # constraints_mat
        # truncation control does truncation  and thus controls for smaller number of cues
        # but does not do the critical manipulation:
        # introducing idiosyncrasies consistent only WITHIN each category
        # e.g. if truncate=0.5, only allow 50% of 1s to actually be 1s
        if self.params.truncate_control == 'col':
            # each column has a different constraint
            constraints_mat = np.random.choice([1, -1],
                                               p=(truncate_prob, 1 - truncate_prob),
                                               size=np.shape(self.untruncated_legals_mat))
        elif self.params.truncate_control == 'mat':
            # all columns have the same constraint
            template = np.random.choice([1, -1], p=(truncate_prob, 1 - truncate_prob), size=self.num_yws)
            constraints_mat = np.expand_dims(template, axis=1).repeat(self.num_xws, axis=1)
        elif self.params.truncate_control == 'none':
            # constraints are the same within basic level categories, but not across
            constraints_mat = np.zeros_like(self.untruncated_legals_mat)
            for cat, cat_col_ids in cat2col_ids.items():
                template = np.random.choice([1, -1], p=(truncate_prob, 1 - truncate_prob), size=self.num_yws)
                constraints_mat[:, cat_col_ids] = np.expand_dims(template, axis=1).repeat(len(cat_col_ids), axis=1)
        else:
            raise AttributeError('Invalid arg to "truncate_control".')
        # truncate
        assert np.shape(constraints_mat) == np.shape(self.untruncated_legals_mat)
        res = np.zeros_like(self.untruncated_legals_mat)
        for col_id in range(self.num_xws):
            old_col = self.untruncated_legals_mat.copy()[:, col_id]
            new_col = constraints_mat[:, col_id]
            #
            res[:, col_id] = [self.params.truncate_sign if old == new == self.params.truncate_sign
                              else -self.params.truncate_sign
                              for old, new in zip(old_col, new_col)]
        print('mean of legals_mat={:.2f}'.format(np.mean(res)))
        assert np.count_nonzero(res) == np.size(res)
        return res

    def plot_heatmap(self, mat, title=None):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=None)
        # heatmap
        print('Plotting heatmap...')
        if title is None:
            title = 'Legals Matrix'.format(self.params.mutation_prob)
        plt.title(title)
        ax.imshow(mat,
                  aspect='equal',
                  # cmap=plt.get_cmap('jet'),
                  cmap='Greys',
                  interpolation='nearest')
        ax.set_xlabel('Probe words')
        ax.set_ylabel('Context words')
        # xticks
        num_cols = len(mat.T)
        ax.set_xticks(np.arange(num_cols))
        ax.xaxis.set_ticklabels([])
        # yticks
        num_rows = len(mat)
        ax.set_yticks(np.arange(num_rows))
        ax.yaxis.set_ticklabels([])
        # remove ticklines
        lines = (ax.xaxis.get_ticklines() +
                 ax.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        plt.show()

    # ///////////////////////////////////////////////////////////////////////// probes

    def make_z(self, method='single', metric='euclidean'):
        # careful: idx1 and idx2 in res are not integers (they are floats)
        corr_mat = to_corr_mat(self.untruncated_legals_mat)
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
        # find cluster (identified by idx1 and idx2) with num_probes nodes beneath it
        for row in self.z:
            idx1, idx2, dist, count = row
            if count == self.params.num_probes:
                break
        else:
            raise RuntimeError('Did not find any cluster with count={}'.format(self.params.num_probes))
        # get probes - tree structure is preserved in order of how probes are retrieved from tree
        res = []
        for idx in [idx1, idx2]:
            self.get_all_probes_in_tree(res, self.z, idx)
        print('Collected {} probes'.format(len(res)))
        assert set(self.x_words) == set(res)
        return res

    def make_probe2cat(self, num_cats):
        print('Assigning probes to {} categories'.format(num_cats))
        num_members = self.params.num_probes / num_cats
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

    # /////////////////////////////////////////////////////////////// sequences

    def sample_from_hierarchical_diffusion(self, node0):
        """the higher the change probability (e),
         the less variance accounted for by higher-up nodes"""
        nodes = [node0]
        for level in range(self.params.num_levels):
            candidate_nodes = nodes * self.params.num_descendants
            s = len(candidate_nodes)
            mutation_prob = 0 if level >= self.params.stop_mutation_level else self.params.mutation_prob
            nodes = [node if p else -node
                     for node, p in zip(candidate_nodes,
                                        np.random.binomial(n=1, p=1 - mutation_prob, size=s))]
        return nodes

    def make_sequences_mat(self):
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
        pool = mp.Pool(processes=num_processes)
        # make sequences
        if len(self.legals_mats) == 1:
            num_chunks = num_processes
            legals_mats = self.legals_mats * num_chunks
        else:
            num_chunks = len(self.legals_mats)
            legals_mats = self.legals_mats
        print('num legals_mats={}'.format(num_chunks))
        print('num num_chunks={}'.format(num_chunks))
        num_seqs_in_chunk = self.params.num_seqs // num_chunks
        results = [pool.apply_async(
            make_sequences_chunk,
            args=(self.x_words, self.y_words, num_seqs_in_chunk, legals_mat)) for legals_mat in legals_mats]
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
            probe_acts2 = self.untruncated_legals_mat[:, [self.x_words.index(p) for p in probes]].T
            ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
            print('num_cats={} input-data col-wise ba={:.3f}'.format(num_cats, ba2))
            res[num_cats] = ba2
        return res

    def plot_tree(self, num_cats):
        probe2color = self.num_cats2probe2color[num_cats]
        link2color = {}
        for i, i12 in enumerate(self.z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(self.z)
                      else probe2color[self.x_words[x.astype(int)]]
                      for x in i12)
            link2color[i + 1 + len(self.z)] = c1 if c1 == c2 else 'grey'
        # plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
        dendrogram(self.z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        ax.set_xticklabels([])
        plt.show()
