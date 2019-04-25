from cytoolz import itertoolz

from treetransitions.params import DefaultParams, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

DefaultParams.num_tokens = [1 * 10 ** 6]
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

    # n-grams
    ngrams = list(itertoolz.sliding_window(params.max_ngram_size + 1, toy_data.tokens))
    num_ngrams = len(set(ngrams))
    print('num_ngrams={:,}'.format(num_ngrams))
    print('------------------------------------------------------------')