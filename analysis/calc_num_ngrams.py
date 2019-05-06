import numpy as np

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 6]
Params.num_cats_list = [[NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]

    # n-grams
    ngram_size = toy_data.id_sequences_mat.shape[1]
    unique_ngrams = np.unique(toy_data.id_sequences_mat, axis=0)
    num_ngrams = len(unique_ngrams)
    print('num_{}-grams={:,}'.format(ngram_size, num_ngrams))
    print('------------------------------------------------------------')