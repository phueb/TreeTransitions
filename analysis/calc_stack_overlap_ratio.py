from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from treetransitions.hierarchical_data_utils import make_tokens, make_probe_data, calc_ba, make_vocab, make_legal_mats
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals

"""
the stack-overlap ratio is a ratio of two counts: stack and overlap.
for each category i,
for each word j,
define stack as the number of types the word j follows members of the category i,
define overlap as the total number of times word j occurs in the input

note: this ratio is only informative using toy data where each legal has the same chance of occurring after
any vocab word. if, for example "cream" followed "ice" frequently, this would increase "stack", 
but this is misleading because "cream" might not follow other dessert words.
"stack" is supposed to be sensitive to the extent to which legals are shared across cat members,
but in this case, "cream" just occurs frequently following a single member of the category.
this would inflate the s-o ratio.
"""


NUM_CATS = 32


params = ObjectView(list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'})[0])
params.parent_count = 512
params.num_tokens = 1 * 10 ** 5
params.num_levels = 10
params.e = 0.2

params.truncate_list = [0.5, 0.6]

vocab, word2id = make_vocab(params.num_descendants, params.num_levels)

raise SystemExit('debugging')

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
    num_after_cats = []  # "stack"
    num_totals = []  # "overlap"
    for word in vocab:
        num_after_cat = word2cat2count[word][cat]
        num_total = word2count[word] + 1
        ratio = num_after_cat / num_total
        if num_after_cat > 0:  # only interested in words that influence category
            num_after_cats.append(num_after_cat)
            num_totals.append(num_total)
    #
    ratios = [stack / overlap for stack, overlap in zip(num_after_cats, num_totals)]
    cat_mean_ratio = np.mean(ratios).round(3)
    cat_var_ratio = np.var(ratios).round(5)
    #
    cat_mean_ratios.append(cat_mean_ratio)
    cat_var_ratios.append(cat_var_ratio)
    print(cat, cat_mean_ratio, cat_var_ratio, len(num_totals), 'max={}'.format(np.max(num_after_cats)))


print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))