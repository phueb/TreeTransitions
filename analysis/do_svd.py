import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import linalg as slinalg
from scipy import sparse
import bisect
from sklearn.metrics.pairwise import cosine_similarity
from cytoolz import itertoolz

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_vocab, make_legal_mats
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


NUM_PCS = 1023
STATISTIC = 'rank-diff'
NGRAM_SIZE = 1
BINARY = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (10, 4)
DPI = None

# /////////////////////////////////////////////////////////////////// make toy data

TRUNCATE_SIZE = 1
NUM_CATS = 32


params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_tokens = 1 * 10 ** 6
params.num_levels = 10
params.e = 0.2
params.truncate_num_cats = NUM_CATS
params.truncate_list = [0.0, 0.1]
truncate_control = [False]  # TODO careful here

vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

# make underlying hierarchical structure
size2word2legals, ngram2legals_mat = make_legal_mats(
    vocab, params.num_descendants, params.num_levels, params.mutation_prob, params.max_ngram_size)

# probes_data
num_cats2word2sorted_legals = {}
print('Getting {} categories with parent_count={}...'.format(NUM_CATS, params.parent_count))
legals_mat = ngram2legals_mat[params.structure_ngram_size]
probes, probe2cat, word2sorted_legals = make_probe_data(
    vocab, size2word2legals[TRUNCATE_SIZE], legals_mat, NUM_CATS, params.parent_count, params.truncate_control)
print('Collected {} probes'.format(len(probes)))
# check probe sim
probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
print('input-data row-wise ba={:.3f}'.format(ba1))
print('input-data col-wise ba={:.3f}'.format(ba2))
print()
num_cats2word2sorted_legals[NUM_CATS] = word2sorted_legals


# sample tokens
tokens = make_tokens(vocab, size2word2legals, num_cats2word2sorted_legals[params.truncate_num_cats],
                     params.num_tokens, params.max_ngram_size, params.truncate_list)


# ///////////////////////////////////////////////////////////////////


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens_part = tokens[start:end]
    # types
    types = sorted(set(tokens))
    num_types = len(types)
    type2id = {t: n for n, t in enumerate(types)}
    # ngrams
    ngrams = list(itertoolz.sliding_window(NGRAM_SIZE, tokens_part))
    ngram_types = sorted(set(ngrams))
    num_ngram_types = len(ngram_types)
    ngram2id = {t: n for n, t in enumerate(ngram_types)}
    # make sparse matrix (types in rows, ngrams in cols)
    shape = (num_types, num_ngram_types)
    print('Making In-Out matrix with shape={}...'.format(shape))
    data = []
    row_ids = []
    cold_ids = []
    mat_loc2freq = {}  # to keep track of number of ngram & type co-occurence
    for n, ngram in enumerate(ngrams[:-NGRAM_SIZE]):
        # row_id + col_id
        col_id = ngram2id[ngram]
        next_ngram = ngrams[n + 1]
        next_type = next_ngram[-1]
        row_id = type2id[next_type]
        # freq
        try:
            freq = mat_loc2freq[(row_id, col_id)]
        except KeyError:
            mat_loc2freq[(row_id, col_id)] = 1
            freq = 1
        else:
            mat_loc2freq[(row_id, col_id)] += 1
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(1 if BINARY else freq)
    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (row_ids, cold_ids)), shape=(num_types, num_ngram_types))
    return res, types


def calc_cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


def calc_some_measure(exp_ids, ref_ids, u_col):
    ref_loadings = []
    exp_loadings = []
    # collect loadings
    for term_id, loading in enumerate(u_col):
        if term_id in exp_ids:
            exp_loadings.append(loading)
        elif term_id in ref_ids:
            ref_loadings.append(loading)
    # compare loadings of reference and experimental group
    assert exp_loadings
    assert ref_loadings
    #
    if STATISTIC == 'cohend':
        return abs(calc_cohend(exp_loadings, ref_loadings))
    elif STATISTIC == 'rank-diff':
        exp_median = np.median(exp_loadings)
        ref_median = np.median(ref_loadings)
        sorted_ref_loadings = sorted(ref_loadings)
        #
        exp_median_rank = bisect.bisect(sorted_ref_loadings, exp_median)
        ref_median_rank = bisect.bisect(sorted_ref_loadings, ref_median)  # this gives max_rank // 2
        #
        res = abs(exp_median_rank - ref_median_rank) / ref_median_rank  # TODO
        # print(exp_median_rank, ref_median_rank, round(res, 2))
        return res
    elif STATISTIC == 't':
        t, pval = stats.ttest_ind(ref_loadings, exp_loadings, equal_var=False)
        return abs(t)
    else:
        raise AttributeError('Invalid arg to "STATISTIC".')


def calc_measure_at_each_pc(pc_mat, types, exp_words, ref_words):
    id2term = {n: t for n, t in enumerate(types)}
    # pre-compute ids at which either reference or experimental word is located
    ids_for_exp = set()
    ids_for_ref = set()
    print('Getting term_ids...')
    for term_id in range(pc_mat.shape[0]):
        term = id2term[term_id]
        if term in exp_words:
            ids_for_exp.add(term_id)
        elif term in ref_words:
            ids_for_ref.add(term_id)
        else:
            pass
    #
    print('num experimental words={}'.format(len(ids_for_exp)))
    print('num reference words={}'.format(len(ids_for_ref)))
    #
    res = []
    print('Calculating some measure for each PC...')
    for pc_id, u_col in enumerate(pc_mat.T):
        assert len(u_col) == pc_mat.shape[0]
        measure = calc_some_measure(ids_for_exp, ids_for_ref, u_col)
        res.append(measure)
    print('Done')
    return res


def plot_comparison(y1, y2, cat, cum, color1='blue', color2='red'):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs of cooc mat (ngram size={})\ncode for the category {}?'.format(
        NUM_PCS, NGRAM_SIZE, cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_ylabel('cumulative abs({})'.format(STATISTIC)
                  if cum else 'abs({})'.format(STATISTIC), fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    adj_alpha = 0.05 / NUM_PCS
    if not cum:
        if STATISTIC == 'cohend':
            ax.set_ylim([0, 3])
        elif STATISTIC == 't':
            df = params.parent_count - 2  # df when equal var is assumed
            p = 1 - adj_alpha
            crit_t = stats.t.ppf(p, df)
            ax.axhline(y=crit_t, color='grey')
        elif STATISTIC == 'rank-diff':
            ax.set_ylim([0, 1])
    else:
        if STATISTIC == 'cohend':
            ax.set_ylim([0, NUM_PCS / 2])
    # plot
    if cum:
        ax.plot(np.cumsum(y1), label=label1, linewidth=2, color=color1)
        ax.plot(np.cumsum(y2), label=label2, linewidth=2, color=color2)
    else:
        ax.plot(y1, label=label1, linewidth=2, color=color1)
        ax.plot(y2, label=label2, linewidth=2, color=color2)
    ax.legend(loc='upper left', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_difference(y, name):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Does first partition represent semantic categories in earlier PCs?\n'
              '(data represented using {}-grams'.format(NGRAM_SIZE), fontsize=AX_FONTSIZE)
    ax.set_ylabel('diff (cumulative abs({}))'.format(STATISTIC), fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    max_y = np.max(np.abs(y))
    ax.set_ylim([-max_y, max_y])
    # plot
    ax.axhline(y=0, color='grey')
    ax.plot(y, linewidth=2)
    plt.tight_layout()
    plt.show()


midpoint_loc = params.num_tokens // 2
start1, end1 = 0, midpoint_loc // 1
start2, end2 = params.num_tokens - end1, params.num_tokens

# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'tokens between\n{:,} & {:,}'.format(start2, end2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(start1, end1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(start2, end2)

# pca1
print('Fitting PCA 1 ...')
sparse_in_out_corr_mat1 = sparse.csr_matrix(in_out_corr_mat1).asfptype()
u, _, _ = slinalg.svds(sparse_in_out_corr_mat1, k=NUM_PCS)
u1 = u[:, :NUM_PCS]
print(u1.shape)

# pca2
print('Fitting PCA 2 ...')
sparse_in_out_corr_mat2 = sparse.csr_matrix(in_out_corr_mat2).asfptype()
u, _, _ = slinalg.svds(sparse_in_out_corr_mat2, k=NUM_PCS)
u2 = u[:, :NUM_PCS]
print(u2.shape)


# collect y across categories
ys1 = []
ys2 = []
cats = set(probe2cat.values())
cat_probe_list_dict = {cat: [w for w in vocab if probe2cat[w] == cat] for cat in cats}
for cat in cats:
    # y
    exp_words = cat_probe_list_dict[cat]
    ref_words = [w for w in vocab if w not in exp_words]
    y1 = calc_measure_at_each_pc(u1, types1, exp_words, ref_words)
    y2 = calc_measure_at_each_pc(u2, types2, exp_words, ref_words)
    #  collect
    ys1.append(y1)
    ys2.append(y2)

# aggregate plot
plot_comparison(np.mean(ys1, axis=0), np.mean(ys2, axis=0), 'all categories', cum=True)
#
cum1 = np.cumsum(np.mean(ys1, axis=0))
cum2 = np.cumsum(np.mean(ys2, axis=0))
diff = cum1 - cum2
plot_difference(diff, 'all categories')