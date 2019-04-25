import numpy as np

from treetransitions.params import DefaultParams, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals

"""
the coverage ratio is a ratio of two counts: A & B
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
this would inflate the coverage ratio.

note: coverage indirectly measures the degree to which contexts coverage across members of a category.
a more direct measurement might take into consideration the number of category members which occur with a context,
because "cream" might occur frequently with "ice" without occurring frequently with other members of the same category
in which "ice" belongs 

note: the coverage ratio approximates 1/NUM_CATS (when truncation is small)
"""

NUM_CATS = 32

DefaultParams.num_tokens = [2 * 10 ** 6]
DefaultParams.num_cats_list = [[NUM_CATS]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.1, 0.1], [1.0, 1.0]]


for param2vals in list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]

    # init
    print('Counting...')
    cats = set(probe2cat.values())
    word2cat2count = {w: {cat: 0 for cat in cats} for w in toy_data.vocab}
    word2noncat_count = {t: 0 for t in toy_data.vocab}
    word2count = {t: 0 for t in toy_data.vocab}
    for n, token in enumerate(toy_data.tokens[1:]):
        prev_token = toy_data.tokens[n-1]
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

    print('Calculating coverage ratios...')
    cat_mean_ratios = []
    cat_var_ratios = []
    for cat in cats:
        num_after_cats = []
        num_totals = []
        for word in toy_data.vocab:
            num_after_cat = word2cat2count[word][cat]
            num_total = word2count[word] + 1
            ratio = num_after_cat / num_total
            if num_after_cat > 0:  # only interested in words that influence category
                num_after_cats.append(num_after_cat)
                num_totals.append(num_total)
        #
        ratios = [a / b for a, b in zip(num_after_cats, num_totals)]
        cat_mean_ratio = np.mean(ratios)
        cat_var_ratio = np.var(ratios)
        #
        cat_mean_ratios.append(cat_mean_ratio)
        cat_var_ratios.append(cat_var_ratio)
        print(cat, cat_mean_ratio.round(3), cat_var_ratio.round(5), len(num_totals), 'max={}'.format(np.max(num_after_cats)))

    print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))
    print(1 / NUM_CATS)
    print('------------------------------------------------------')