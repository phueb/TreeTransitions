import numpy as np
import pyprind
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from bayes_opt import BayesianOptimization
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp
from cytoolz import itertoolz
from scipy.spatial.distance import pdist
from matplotlib.colors import to_hex


NUM_PROCESSES = 4


def generate_tokens_from_zipfian(words, num_tokens, oov='OOV'):  # TODO unused
    num_vocab = len(words)
    res = [words[i] if i < num_vocab else oov for i in np.random.zipf(2, num_tokens)]
    return res


def get_all_probes_in_tree(vocab, res1, z, node_id):
    """
    # z should be the result of linkage,which returns an array of length n - 1
    # giving you information about the n - 1 cluster merges which it needs to pairwise merge n clusters.
    # Z[i] will tell us which clusters were merged in the i-th iteration.
    # a row in z is: [idx1, idx2, distance, count]
    # all idx >= len(X) actually refer to the cluster formed in Z[idx - len(X)]
    """
    try:
        p = vocab[node_id.astype(int)]
    except IndexError:  # in case idx does not refer to leaf node (then it refers to cluster)
        new_node_id1 = z[node_id.astype(int) - len(vocab)][0]
        new_node_id2 = z[node_id.astype(int) - len(vocab)][1]
        get_all_probes_in_tree(vocab, res1, z, new_node_id1)
        get_all_probes_in_tree(vocab, res1, z, new_node_id2)
    else:
        res1.append(p)


def make_probe_data(vocab, word2id, legals_mat, num_cats, parent_count,
                    method='single', metric='euclidean', plot=True):
    """
    make categories from hierarchically organized data.
    """
    num_vocab = len(vocab)
    num_members = parent_count / num_cats
    assert legals_mat.shape == (num_vocab, num_vocab)
    assert num_cats % 2 == 0
    assert num_members.is_integer()
    num_members = int(num_members)
    # get z - careful: idx1 and idx2 in z are not integers (they are floats)
    corr_mat = to_corr_mat(legals_mat)
    z = linkage(corr_mat, metric=metric, method=method)  # need to cluster correlation matrix otherwise result is messy
    # find cluster (identified by idx1 and idx2) with parent_count nodes beneath it
    for row in z:
        idx1, idx2, dist, count = row
        if count == parent_count:
            break
    else:
        raise RuntimeError('Did not find any cluster with count={}'.format(parent_count))
    # get probes - tree structure is preserved in order of how probes are retrieved from tree
    retrieved_probes = []
    for idx in [idx1, idx2]:
        get_all_probes_in_tree(vocab, retrieved_probes, z, idx)
    # split into categories
    probe2cat = {}
    probes = []
    cat_probes_list = []
    probe2color = {}
    cat2sorted_legals = {}
    cmap = plt.cm.get_cmap('hsv', num_cats + 1)
    for cat_id, cat_probes in enumerate(itertoolz.partition_all(num_members, retrieved_probes)):
        assert len(cat_probes) == num_members
        probe2cat.update({p: cat_id for p in cat_probes})
        probes.extend(cat_probes)
        cat_probes_list.append(cat_probes)
        for p in cat_probes:
            probe2color[p] = to_hex(cmap(cat_id))
        #  get most diagnostic legals for cat
        cols = legals_mat[:, [word2id[p] for p in cat_probes]]
        cat_sorted_legal_ids = np.argsort(cols.sum(axis=1))  # sorted by lowest to highest cat diagnostic-ity
        cat2sorted_legals[cat_id] = [vocab[i] for i in cat_sorted_legal_ids]  # typically almost as large as vocab
    # convert cat2sorted_legals to word2sorted_legals
    non_cat_sorted_legal_ids = np.argsort(legals_mat.sum(axis=1))
    non_cat_sorted_legals = [vocab[i] for i in non_cat_sorted_legal_ids]
    word2sorted_legals = {}
    for word in vocab:
        if word in probes:
            cat = probe2cat[word]
            word2sorted_legals[word] = cat2sorted_legals[cat]
        else:
            word2sorted_legals[word] = non_cat_sorted_legals  # careful: sorted in ascending order - truncate from end
    #
    if plot:

        link2color = {}
        for i, i12 in enumerate(z[:, :2].astype(int)):
            c1, c2 = (link2color[x] if x > len(z) else probe2color[vocab[x.astype(int)]]
                      for x in i12)
            link2color[i + 1 + len(z)] = c1 if c1 == c2 else 'grey'

        colors_it = iter([to_hex(cmap(i)) for i in range(num_cats)])
        fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
        dg = dendrogram(z, ax=ax, color_threshold=None, link_color_func=lambda i: link2color[i])
        reordered_vocab = np.asarray(vocab)[dg['leaves']]
        ax.set_xticklabels([w if w in probes else '' for w in reordered_vocab], fontsize=6)
        for cat_probes in cat_probes_list:
            color = next(colors_it)
            for i in range(num_vocab):
                if reordered_vocab[i] in cat_probes:
                    ax.get_xticklabels()[i].set_color(color)
        plt.show()
        #
        # clustered_corr_mat = corr_mat[dg['leaves'], :]
        # clustered_corr_mat = clustered_corr_mat[:, dg['leaves']]
        # plot_heatmap(clustered_corr_mat, [], [], dpi=None)
        # plot_heatmap(cluster(data_mat), [], [], dpi=None)
    return probes, probe2cat, word2sorted_legals


def sample_from_hierarchical_diffusion(node0, num_descendants, num_levels, e):
    """the higher the change probability (e),
     the less variance accounted for by higher-up nodes"""
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if p else -node for node, p in zip(candidate_nodes,
                                                         np.random.binomial(n=2, p=1 - e, size=len(candidate_nodes)))]
    return nodes


def make_chunk(chunk_id, size2word2legals, word2sorted_legals, vocab, num_start, chunk_size, truncate,
               random_interval=np.nan):
    print('\nMaking tokens chunk with truncate={}'.format(truncate)) if chunk_id % NUM_PROCESSES == 0 else None
    #
    tokens_chunk = np.random.choice(vocab, size=num_start).tolist()  # prevents indexError at start
    pbar = pyprind.ProgBar(chunk_size) if chunk_id % NUM_PROCESSES == 0 else None
    for loc in range(chunk_size):
        # append random word to break structure into pseudo-sentences
        if loc % random_interval == 0:
            new_token = np.random.choice(vocab, size=1).item()
            tokens_chunk.append(new_token)
            continue
        # append word which is constrained by hierarchical structure
        else:
            # get words which are legal to come next
            legals_set = set(vocab)
            for size, word2legals in size2word2legals.items():
                previous_token = tokens_chunk[-size]
                sorted_legals = word2sorted_legals[previous_token]
                num_truncated = int(len(sorted_legals) * truncate)
                #
                legals = word2legals[previous_token]
                #
                legals_set.intersection_update(legals)
                legals_set.intersection_update(sorted_legals[-num_truncated:])  # truncate from end TODO test
            # sample from legals
            try:
                new_token = np.random.choice(list(legals_set), size=1, p=None).item()
            except ValueError:  # no legals
                raise RuntimeError('No legal next word available. Increase mutation_prob')
            # collect
            tokens_chunk.append(new_token)
        pbar.update() if chunk_id % NUM_PROCESSES == 0 else None
    return tokens_chunk


def make_vocab(num_descendants, num_levels):
    num_vocab = num_descendants ** num_levels
    vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
    word2id = {word: n for n, word in enumerate(vocab)}
    return vocab, word2id


def make_legal_mats(vocab, num_descendants, num_levels, mutation_prob, max_ngram_size):
    # ngram2legals_mat - each row specifies legal next words (col_words)
    ngram_sizes = range(1, max_ngram_size + 1)
    word2node0 = {}
    num_vocab = len(vocab)
    ngram2legals_mat = {ngram: np.zeros((num_vocab, num_vocab), dtype=np.int) for ngram in ngram_sizes}
    print('Making hierarchical dependency structure...')
    for ngram_size in ngram_sizes:
        for row_id, word in enumerate(vocab):
            node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
            word2node0[word] = node0
            ngram2legals_mat[ngram_size][row_id, :] = sample_from_hierarchical_diffusion(
                node0, num_descendants, num_levels, mutation_prob)  # this row specifies col_words which predict the row_word
    print('Done')
    # collect legal next words for each word at each distance - do this once to speed calculation of tokens
    # whether -1 or 1 determines legality depends on node0 - otherwise half of words are never legal
    size2word2legals = {}
    for ngram_size in ngram_sizes:
        legals_mat = ngram2legals_mat[ngram_size]
        word2legals = {}
        for col_word, col in zip(vocab, legals_mat.T):  # col contains information about which row_words come next
            legals = [w for w, val in zip(vocab, col) if val == word2node0[w]]
            word2legals[col_word] = np.random.permutation(legals)  # to ensure truncation affects each word differently
        size2word2legals[ngram_size] = word2legals
    return size2word2legals, ngram2legals_mat


def make_tokens(vocab, size2word2legals, word2sorted_legals,
                num_tokens, max_ngram_size, truncate_list, num_chunks=32):
    """
    generate text by adding one word at a time to a list of words.
    each word is constrained by the legals matrices - which are hierarchical -
    and determine the legal successors for each word in the vocabulary.
    there is one legal matrix for each ngram (in other words, for each distance up to max_ngram_size)
    the set of legal words that can follow a word is the intersection of legal words dictated by each legals matrix.
    the size of the intersection is approximately the same for each word, and a word is sampled uniformly from this set
    the size of the intersection is the best possible perplexity a model can achieve,
    because perplexity indicates the number of choices from a random uniformly distributed set of choices
    """
    # make tokens - in parallel
    pool = mp.Pool(processes=NUM_PROCESSES)
    min_truncate, max_truncate = truncate_list
    truncate_steps = np.linspace(min_truncate, max_truncate, num_chunks + 2)[1:-1]
    chunk_size = num_tokens // num_chunks
    results = [pool.apply_async(
        make_chunk,
        args=(chunk_id, size2word2legals, word2sorted_legals, vocab, max_ngram_size,
              chunk_size, truncate_steps[chunk_id]))
        for chunk_id in range(num_chunks)]
    tokens = []
    print('Creating tokens from hierarchical dependency structure...')
    try:
        for res in results:
            tokens_chunk = res.get()
            print('\nnum types in tokens_chunk={}'.format(len(set(tokens_chunk))))
            tokens += tokens_chunk
        pool.close()
    except KeyboardInterrupt:
        pool.close()
        raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
    print('Done')
    return tokens


def calc_ba(probe_sims, probes, probe2cat, num_opt_init_steps=1, num_opt_steps=10):
    def calc_signals(_probe_sims, _labels, thr):  # vectorized algorithm is 20X faster
        probe_sims_clipped = np.clip(_probe_sims, 0, 1)
        probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
        predictions = np.zeros_like(probe_sims_clipped_triu, int)
        predictions[np.where(probe_sims_clipped_triu > thr)] = 1
        #
        tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
        tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
        fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
        fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
        return tp, tn, fp, fn

    # gold_mat
    if not len(probes) == probe_sims.shape[0] == probe_sims.shape[1]:
        raise RuntimeError(len(probes), probe_sims.shape[0], probe_sims.shape[1])
    num_rows = len(probes)
    num_cols = len(probes)
    gold_mat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        probe1 = probes[i]
        for j in range(num_cols):
            probe2 = probes[j]
            if probe2cat[probe1] == probe2cat[probe2]:
                gold_mat[i, j] = 1

    # define calc_signals_partial
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, probe_sims, labels)

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    sims_mean = np.mean(probe_sims).item()
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    bo = BayesianOptimization(calc_probes_ba, {'thr': (0.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=num_opt_init_steps, n_iter=num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = calc_probes_ba(best_thr)
    res = np.mean(results)
    return res


def to_corr_mat(data_mat):
    zscored = stats.zscore(data_mat, axis=0, ddof=1)
    res = np.matmul(zscored.T, zscored)  # it matters which matrix is transposed
    return res


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


def cluster(mat, original_row_words=None, original_col_words=None):
    print('Clustering...')
    #
    lnk0 = linkage(pdist(mat))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)
    z = mat[dg0['leaves'], :]  # reorder rows
    #
    lnk1 = linkage(pdist(mat.T))
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