from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

from treetransitions.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32


params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 512
params.num_tokens = 1 * 10 ** 5
params.num_levels = 10
params.e = 0.2

params.truncate_list = [0.0, 1.0]


# make tokens with hierarchical n-gram structure
vocab, tokens, ngram2legals_mat = make_data(
    params.num_tokens, params.legals_distribution, params.max_ngram_size,
    params.num_descendants, params.num_levels, params.mutation_prob, params.truncate_list)
num_vocab = len(vocab)
num_types_in_tokens = len(set(tokens))
word2id = {word: n for n, word in enumerate(vocab)}
token_ids = [word2id[w] for w in tokens]
print()
print('num_vocab={}'.format(num_vocab))
print('num types in tokens={}'.format(num_types_in_tokens))
if not num_types_in_tokens == num_vocab:
    print('Not all types ({}/{} were found in tokens.'.format(num_types_in_tokens, num_vocab))

# probes_data
num_cats2probes_data = {}
num_cats2max_ba = {}
print('Getting {} categories with parent_count={}...'.format(NUM_CATS, params.parent_count))
legals_mat = ngram2legals_mat[params.structure_ngram_size]
probes, probe2cat = make_probe_data(legals_mat, vocab, NUM_CATS, params.parent_count, plot=False)
num_cats2probes_data[NUM_CATS] = (probes, probe2cat)
c = Counter(tokens)
for p in probes:
    # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
    if c[p] < 10:
        print('WARNING: "{}" occurs only {} times'.format(p, c[p]))
print('Collected {} probes'.format(len(probes)))
# check probe sim
probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
print('input-data row-wise ba={:.3f}'.format(ba1))
print('input-data col-wise ba={:.3f}'.format(ba2))
print()


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

print('Calculating ratios...')
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
    ratios = [n/d for n, d in zip(num_after_cats, num_totals)]
    cat_mean_ratio = np.mean(ratios).round(3)
    cat_var_ratio = np.var(ratios).round(5)
    #
    cat_mean_ratios.append(cat_mean_ratio)
    cat_var_ratios.append(cat_var_ratio)
    print(cat, cat_mean_ratio, cat_var_ratio, len(num_totals), 'max={}'.format(np.max(num_after_cats)))


print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))