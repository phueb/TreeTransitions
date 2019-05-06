import numpy as np

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals

"""
the coverage ratio is a ratio of two counts: A & B
for each category i,
for each word j,
define Aij as the number of members of category i that word j occurs at least once after
define Bij as the number of vocabulary words that word j occurs at least once after
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

note: the coverage ratio converges on 1/NUM_CATS (as num_seqs increases)
"""

NUM_CATS = 32

Params.num_seqs = [2 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


def make_bigram_count_mat(id_seqs_mat, num_vocab):
    assert id_seqs_mat.shape[1] == 2  # works with bi-grams only
    print('Making In-Out matrix with shape={}...'.format(id_seqs_mat.shape))
    res = np.zeros((num_vocab, num_vocab))
    for row_id, col_id in zip(id_seqs_mat[:, 0], id_seqs_mat[:, 1]):
        # freq
        res[row_id, col_id] += 1
    return res


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cats = set(probe2cat.values())

    bigram_count_mat = make_bigram_count_mat(toy_data.id_sequences_mat, toy_data.num_vocab)

    cat_mean_ratios = []
    cat_var_ratios = []
    for cat in cats:
        cat_probes = [p for p in toy_data.probes if probe2cat[p] == cat]
        cat_coverages = []
        vocab_coverages = []
        for col in bigram_count_mat.T:  # iterate over contexts
            num_after_cat = [count if toy_data.vocab[word_id] in cat_probes else 0
                             for word_id, count in enumerate(col)]
            num_after_vocab = [count for word_id, count in enumerate(col)]
            cat_coverage = np.count_nonzero(num_after_cat)  # how often context occurs after category
            vocab_coverage = np.count_nonzero(num_after_vocab)  # how often context occurs after any word
            #
            cat_coverages.append(cat_coverage)
            vocab_coverages.append(vocab_coverage)

        ratios = [(a + 0.001) / (b + 0.001) for a, b in zip(cat_coverages, vocab_coverages)]
        cat_mean_ratio = np.mean(ratios)
        cat_var_ratio = np.var(ratios)
        #
        cat_mean_ratios.append(cat_mean_ratio)
        cat_var_ratios.append(cat_var_ratio)
        print(cat, cat_mean_ratio.round(3), cat_var_ratio.round(5))

    print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))
    print(1 / NUM_CATS)
    print('------------------------------------------------------')