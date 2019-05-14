import numpy as np
import pyprind
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib.colors import to_hex
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.utils import to_corr_mat, calc_ba
from treetransitions import config


def make_sequences_chunk(x_words, y_words, num_seqs, legals_mat):
    print('\nMaking sequences chunk...')

    print(legals_mat.shape)

    # xw2yws
    xw2yws = {}
    for xw, col in zip(x_words, legals_mat.T):
        yws = [yw for yw, val in zip(y_words, col) if val == 1]
        xw2yws[xw] = yws
    #  make seqs
    seq_size = 2
    res = np.random.choice(x_words, size=(num_seqs, seq_size))
    for seq in res:
        # get words which are legal to come next - y_words
        xw = seq[0]
        legal_yws = xw2yws[xw]
        # sample from legal y_words
        yw = np.random.choice(list(legal_yws), size=1, p=None).item()
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
        assert min(self.params.num_cats_list) == self.params.min_num_cats
        self.x_words = self.make_x_words()  # probes
        self.y_words = self.make_y_words()  # non-probes (or context words)
        self.num_yws = len(self.y_words)
        self.num_xws = len(self.x_words)
        #
        self.vocab = list(set(self.y_words + self.x_words))
        self.num_vocab = len(self.vocab)  # used by rnn
        self.word2id = {word: n for n, word in enumerate(self.vocab)}
        #
        self.num_cats2xw2cat = {num_cats: self.make_xw2cat(num_cats) for num_cats in params.num_cats_list}
        self.xw2cat = self.num_cats2xw2cat[self.params.min_num_cats]
        self.yw2cat = self.make_yw2cat()
        #
        self.num_expansions = int(np.log2(self.params.num_probes) - np.log2(self.params.min_num_cats))
        self.template_mat = self.make_template_mat()
        self.legals_mats = list(self.make_legals_mats(self.template_mat))
        self.full_legals_mat = self.legals_mats[-1]
        # ba
        self.num_cats2max_ba = self.make_num_cats2max_ba() if max_ba else None
        # plot
        self.num_cats2cmap = {num_cats: plt.cm.get_cmap('hsv', num_cats + 1) for num_cats in params.num_cats_list}
        self.num_cats2probe2color = {num_cats: self.make_xw2color(num_cats) for num_cats in params.num_cats_list}
        for num_cats in self.params.num_cats_list:
            self.z = self.make_z() if config.Eval.plot_tree else None
            self.plot_tree(num_cats) if config.Eval.plot_tree else None
        # pot legals_mat
        if config.Eval.plot_legals_mat:
            for legal_mat in self.legals_mats:
                self.plot_legals_mat(legal_mat)
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

    def make_xw2cat(self, num_cats):
        num_members = self.params.num_probes // num_cats
        print('num xws in each cat={}'.format(num_members))
        res = {xw: cat for xw, cat in zip(self.x_words, np.repeat(np.arange(num_cats), num_members))}
        return res

    def make_yw2cat(self):
        num_members = self.params.num_contexts // self.params.min_num_cats
        print('num yws in each cat={}'.format(num_members))
        res = {yw: cat for yw, cat in zip(self.y_words, np.repeat(np.arange(self.params.min_num_cats), num_members))}
        return res

    # /////////////////////////////////////////////////////////////// legals

    def make_template_mat(self):
        """
        make a binary vector for each word (same for category members)
        which will be used as input to branching diffusion
        """
        res = []
        print('Making template_mat...')
        for n, yw in enumerate(self.y_words):
            template = -np.ones(self.params.min_num_cats)
            template *= [1 if binom else -1 for binom in np.random.binomial(
                n=1, p=1 - self.params.template_noise, size=len(template))]  # adding noise
            cat_id = self.yw2cat[yw]
            template[cat_id] = 1  # do this after adding noise
            #
            res.append(template)
        return np.vstack(res)

    def make_legals_mats(self, template_mat):  # TODO test
        expanded = template_mat
        for _ in range(self.num_expansions):
            expanded = self.branching_diffusion(expanded)
            yield self.complete_branching(expanded)

    def plot_legals_mat(self, mat):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=None)
        # heatmap
        print('Plotting heatmap...')
        plt.title('template_noise={}\nmutation_prob={}'.format(
            self.params.template_noise, self.params.mutation_prob))
        ax.imshow(mat,
                  aspect='equal',
                  cmap=plt.get_cmap('jet'),
                  interpolation='nearest')
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

    def branching_diffusion(self, template_mat):
        """
        for a given context_word, create binary vector representing which probes it is allowed to follow
        """
        num_descendants = 2
        res = []
        for row in template_mat:
            rep = np.repeat(row, num_descendants)
            expanded = rep * [1 if b else -1
                              for b in np.random.binomial(n=1, p=1-self.params.mutation_prob, size=len(rep))]
            res.append(expanded)
        return np.vstack(res)

    def complete_branching(self, template_mat):
        num_descendants = 2
        res = []
        for row in template_mat:
            while True:
                if len(row) >= self.params.num_probes:
                    break
                row = np.repeat(row, num_descendants)  # no mutation
            res.append(row)
        return np.vstack(res)

    # /////////////////////////////////////////////////////////////// sequences

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
        num_seqs_in_chunk = self.params.num_seqs // self.num_expansions
        results = [pool.apply_async(
            make_sequences_chunk,
            args=(self.x_words, self.y_words, num_seqs_in_chunk, legals_mat)) for legals_mat in self.legals_mats]
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
            probes = self.x_words
            probe_acts2 = self.full_legals_mat[:, [self.x_words.index(p) for p in probes]].T
            ba2 = calc_ba(cosine_similarity(probe_acts2), probes, self.num_cats2xw2cat[num_cats])
            print('input-data col-wise ba={:.3f}'.format(ba2))
            res[num_cats] = ba2
        return res

    def make_z(self, method='single', metric='euclidean'):
        # careful: idx1 and idx2 in res are not integers (they are floats)
        corr_mat = to_corr_mat(self.full_legals_mat)
        res = linkage(corr_mat, metric=metric, method=method)  # need to cluster correlation matrix otherwise messy
        return res

    def make_xw2color(self, num_cats):
        res = {}
        cmap = self.num_cats2cmap[num_cats]
        for xw in self.x_words:
            cat = self.num_cats2xw2cat[num_cats][xw]
            res[xw] = to_hex(cmap(cat))
        return res

    def plot_tree(self, num_cats):
        print('Plotting tree with num_cats={}'.format(num_cats))
        xw2color = self.num_cats2probe2color[num_cats]
        link2color = {}
        for i, i12 in enumerate(self.z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(self.z)
                      else xw2color[self.x_words[x.astype(int)]]
                      for x in i12)
            link2color[i + 1 + len(self.z)] = c1 if c1 == c2 else 'grey'
        # plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
        dendrogram(self.z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        ax.set_xticklabels([])
        plt.show()
