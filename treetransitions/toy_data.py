import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import multiprocessing as mp
from cytoolz import itertoolz
from matplotlib.colors import to_hex
from sklearn.metrics.pairwise import cosine_similarity
from sortedcontainers import SortedDict

from treetransitions.utils import to_corr_mat, calc_ba, cluster
from treetransitions import config


def make_sequences_chunk(num_seqs, name2words, name2legals_mat, structure_prob, corpus_seed):
    np.random.seed(corpus_seed)
    print('Making {} sequences with random seed={}...'.format(num_seqs, corpus_seed))
    # xw2yws
    xw2yws = SortedDict()
    for name, legals_mat in name2legals_mat.items():
        xws, yws = name2words[name]
        assert len(xws) == len(legals_mat.T)
        if name == 'p':
            legal_vals = iter([1] * legals_mat.size)  # this preserves hierarchical structure of probes
        else:
            legal_vals = iter(np.random.choice([1, -1], size=legals_mat.size, p=[structure_prob, 1 - structure_prob]))
        for xw, col in zip(xws, legals_mat.T):
            assert xw not in xw2yws
            xw2yws[xw] = [yw for yw, val in zip(yws, col)    # if-statement is required
                          if val == next(legal_vals)]  # this reduces number of legal sequences, not just hierarchy
    #
    num_legals = np.concatenate([values for values in xw2yws.values()]).size
    print('number of legal sequences={:,}'.format(num_legals))
    #
    seq_size = 2
    res = np.random.choice(xw2yws.keys(), size=(num_seqs, seq_size))
    for seq in res:
        xw = seq[0]
        yw = np.random.choice(xw2yws[xw], size=1, p=None).item()
        seq[-1] = yw
    np.random.seed(None)
    return res


"""
legals_mat: 
each row contains a vector sampled from hierarchical process.
each column (x-word) represents which y_words are allowed to follow the x_word
"""


class ToyData:
    def __init__(self, params, max_ba=True, make_tokens=True):
        self.params = params
        # use SortedDict to prevent non-deterministic corpus generation
        self.name2words = SortedDict({str(n): (self.make_words(str(n) + 'x', num_non_probes),
                                               self.make_words(str(n) + 'y', self.params.num_contexts))
                                      for n, num_non_probes in enumerate(self.params.num_non_probes_list)})
        self.name2words['p'] = (self.make_words('px', self.params.num_probes),
                                self.make_words('py', self.params.num_contexts))
        self.x_words = self.name2words['p'][0]  # probes
        self.y_words = self.name2words['p'][1]  # probe-contexts
        #
        self.vocab = self.make_vocab()
        self.num_vocab = len(self.vocab)  # used by rnn
        self.word2id = {word: n for n, word in enumerate(self.vocab)}
        #
        self.name2legals_mat = SortedDict({name: self.make_legals_mat(name, xws, yws)
                                           for name, (xws, yws) in self.name2words.items()})
        self.probes_legals_mat = self.name2legals_mat['p']
        #
        self.z = self.make_z()
        self.num_cats2probe2cat = {num_cats: self.make_probe2cat(num_cats)
                                   for num_cats in params.num_cats_list}
        # ba
        self.num_cats2max_ba = self.make_num_cats2max_ba() if max_ba else None
        # plot
        self.num_cats2cmap = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1) for num_cats in params.num_cats_list}
        self.num_cats2probe2color = {num_cats: self.make_probe2color(num_cats) for num_cats in params.num_cats_list}
        for num_cats in self.params.num_cats_list:
            self.plot_tree(num_cats) if config.Eval.plot_tree else None
        # plot legals_mat
        if config.Eval.plot_legals_mat:
            for name, legals_mat in self.name2legals_mat.items():
                self.plot_heatmap(legals_mat, name)
        # pot legals_mat correlation matrix
        if config.Eval.plot_corr_mat:
            for name, legals_mat in self.name2legals_mat.items():
                self.plot_heatmap(cluster(to_corr_mat(legals_mat)), name)
        #
        if make_tokens:
            self.word_sequences_mat = self.make_sequences_mat()
            self.num_seqs = len(self.word_sequences_mat)  # divisible by mb_size and num_partitions
            print('shape of word_sequences_mat={}'.format(self.word_sequences_mat.shape))
            self.id_sequences_mat = np.asarray([[self.word2id[w] for w in seq]
                                                for seq in self.word_sequences_mat]).reshape((self.num_seqs, -1))

    @staticmethod
    def make_words(prefix, num):
        vocab = ['{}{}'.format(prefix, i) for i in np.arange(num)]
        return vocab

    def make_vocab(self):
        res = []
        for name, (xws, yws) in self.name2words.items():
            res.extend(xws + yws)
        print('Vocab size={}'.format(len(res)))
        return res

    # /////////////////////////////////////////////////////////////// legals

    def make_legals_mat(self, name, xws, yws):
        """
        each row contains a vector sampled from hierarchical process.
        each column represents which y_words are allowed to follow an x_word (word associated with column)
        """
        np.random.seed(int(name) if name != 'p' else self.params.corpus_seed)
        num_total_nodes = len(self.name2words[name][0])
        res = np.zeros((len(yws), len(xws)), dtype=np.int)
        for row_id, yw in enumerate(yws):
            res[row_id, :] = self.sample_from_hierarchical_diffusion(num_total_nodes)
        one_prob = np.count_nonzero(np.clip(res, 0, 1)) / np.size(res)
        # control for hierarchical structure in non-probes legals_mat
        if not self.params.non_probes_hierarchy and name != 'p':
            print('Making legals_mat for "{}" words without hierarchical structure'.format(name))
            res = np.random.choice([1, -1], size=np.shape(res), p=[one_prob, 1 - one_prob])
            print(res)
        print('"{}" legals_mat: shape={} probability of +1={}'.format(name, res.shape, one_prob))
        np.random.seed(None)
        return res

    def plot_heatmap(self, mat, name):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=None)
        # heatmap
        print('Plotting heatmap...')
        plt.title('Legals Matrix'.format(self.params.mutation_prob))
        ax.imshow(mat,
                  aspect='equal',
                  # cmap=plt.get_cmap('jet'),
                  cmap='Greys',
                  interpolation='nearest')
        ax.set_xlabel('Words in Category "{}"'.format(name))
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
        corr_mat = to_corr_mat(self.probes_legals_mat)
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
        num_xws = len(self.x_words)
        try:
            p = self.x_words[node_id.astype(int)]
        except IndexError:  # in case idx does not refer to leaf node (then it refers to cluster)
            new_node_id1 = z[node_id.astype(int) - num_xws][0]
            new_node_id2 = z[node_id.astype(int) - num_xws][1]
            self.get_all_probes_in_tree(res1, z, new_node_id1)
            self.get_all_probes_in_tree(res1, z, new_node_id2)
        else:
            res1.append(p)

    def make_probe2cat(self, num_cats):
        # find cluster (identified by idx1 and idx2) with num_probes nodes beneath it
        for row in self.z:
            idx1, idx2, dist, count = row
            if count == self.params.num_probes:
                break
        else:
            raise RuntimeError('Did not find any cluster with count={}'.format(self.params.num_probes))
        # get xwords in correct order - tree structure is preserved in order of how probes are retrieved from tree
        ordered_xws = []
        for idx in [idx1, idx2]:
            self.get_all_probes_in_tree(ordered_xws, self.z, idx)
        print('Collected {} probes'.format(len(ordered_xws)))
        assert set(self.x_words) == set(ordered_xws)
        #
        print('Assigning probes to {} categories'.format(num_cats))
        num_members = self.params.num_probes / num_cats
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
        for p in self.x_words:
            cat = self.num_cats2probe2cat[num_cats][p]
            res[p] = to_hex(cmap(cat))
        return res

    # /////////////////////////////////////////////////////////////// sequences

    def sample_from_hierarchical_diffusion(self, num_total_nodes):
        """the higher the change probability (e),
         the less variance accounted for by higher-up nodes"""
        num_descendants = 2
        nodes = [1]
        while True:
            candidate_nodes = nodes * num_descendants
            s = len(candidate_nodes)
            nodes = [node if p else -node
                     for node, p in zip(candidate_nodes,
                                        np.random.binomial(n=1, p=1 - self.params.mutation_prob, size=s))]
            if len(nodes) >= num_total_nodes:
                break
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
        num_chunks = self.params.num_partitions * num_processes
        num_seqs_in_chunk = self.params.num_seqs // num_chunks
        linspace = np.repeat(
            np.linspace(*self.params.structure_probs, self.params.num_partitions, endpoint=True),
            num_processes).round(2)
        print('linspace:')
        print(linspace)
        results = [pool.apply_async(
            make_sequences_chunk,
            args=(num_seqs_in_chunk, self.name2words, self.name2legals_mat, structure_prob, self.params.corpus_seed))
            for structure_prob in linspace]
        chunks = []
        print('Creating sequences...')
        try:
            for n, r in enumerate(results):
                chunk = r.get()
                if n % num_processes == 0:
                    print('num probes in chunk={}'.format(len([w for w in chunk[:, 0] if w in self.x_words])))
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
        print('Shortened num_seqs to {:,} by removing {:,} seqs'.format(num_divisible, num_remainder))
        return stacked[:num_divisible]

    def gen_part_id_seqs(self):
        for part_seq in np.vsplit(self.id_sequences_mat, self.params.num_partitions):
            print('Shape of part_id_seq={}'.format(part_seq.shape))
            yield part_seq

    # //////////////////////////////////////////////////////////// misc

    def make_num_cats2max_ba(self):
        res = {}
        for num_cats in self.params.num_cats_list:
            probes = self.x_words
            probe2cat = self.num_cats2probe2cat[num_cats]
            probe_acts2 = self.probes_legals_mat[:, [self.x_words.index(p) for p in probes]].T
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
