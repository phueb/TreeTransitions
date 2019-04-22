from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_vocab, make_legal_mats
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals

"""
the overlap ratio is a ratio of two counts: A & B
for each category i,
for each word j,
define Aij as the number of times the word j follows members of the category i,
define Bij as the total number of times word j occurs in the input
add all Aij to form A, and add all Bij to form B, then divide to get the  ratio

note: this ratio is only informative using toy data where each legal has the same chance of occurring after
any vocab word. if, for example "cream" followed "ice" frequently, this would increase the numerator, 
but this is misleading because "cream" might not follow other dessert words.
the numerator is supposed to be sensitive to the extent to which legals are shared across cat members,
but in this case, "cream" just occurs frequently following a single member of the category.
this would inflate the overlap ratio.

note: overlap indirectly measures the degree to which contexts overlap across members of a category.
a more direct measurement might take into consideration the number of category members which occur with a context,
because "cream" might occur frequently with "ice" without occurring frequently with other members of the same category
in which "ice" belongs 
"""

TRUNCATE_SIZE = 1
NUM_CATS = 32


params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 1024
params.num_tokens = 1 * 10 ** 5
params.num_levels = 10
params.e = 0.2
params.truncate_num_cats = NUM_CATS

params.truncate_list = [0.1, 0.0]

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

# init
print('Counting...')
cats = set(probe2cat.values())
word2cat2count = {w: {cat: 0 for cat in cats} for w in vocab}
word2noncat_count = {t: 0 for t in vocab}
word2count = {t: 0 for t in vocab}
for n, token in enumerate(tokens[1:]):
    prev_token = tokens[n-1]
    try:
        prev_cat = probe2cat[prev_token]
    except KeyError:
        prev_cat = None
    #
    if prev_cat is None:
        word2noncat_count[token] += 1
    else:
        word2cat2count[token][prev_cat] += 1
    #
    word2count[token] += 1

print('Calculating stack-overlap ratios...')
cat_mean_ratios = []
cat_var_ratios = []
for cat in cats:
    num_after_cats = []
    num_totals = []
    for word in vocab:
        num_after_cat = word2cat2count[word][cat]
        num_total = word2count[word] + 1
        ratio = num_after_cat / num_total
        if num_after_cat > 0:  # only interested in words that influence category
            num_after_cats.append(num_after_cat)
            num_totals.append(num_total)
    #
    ratios = [a / b for a, b in zip(num_after_cats, num_totals)]
    cat_mean_ratio = np.mean(ratios).round(3)
    cat_var_ratio = np.var(ratios).round(5)
    #
    cat_mean_ratios.append(cat_mean_ratio)
    cat_var_ratios.append(cat_var_ratio)
    print(cat, cat_mean_ratio, cat_var_ratio, len(num_totals), 'max={}'.format(np.max(num_after_cats)))


print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))
print(1 / NUM_CATS)

# TODO the overlap ratio approximates 1/NUM_CATS (when truncation is small)