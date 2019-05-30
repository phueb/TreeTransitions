import numpy as np

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

Params.num_seqs = [1 * 10 ** 6]
Params.num_cats_list = [[32]]

sets = []
for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    sp = tuple(params.structure_probs)

    # n-grams
    ngram_size = toy_data.id_sequences_mat.shape[1]
    unique_ngrams = np.unique(toy_data.id_sequences_mat, axis=0)
    num_ngrams = len(unique_ngrams)
    sets.append(set([tuple(ngram) for ngram in unique_ngrams]))
    print('num_{}-grams={:,}'.format(ngram_size, num_ngrams))
    print('------------------------------------------------------------')


updated_ngrams = sets[1].copy()
print(len(updated_ngrams))
updated_ngrams.update(sets[0])
num_updated_ngrams = len(updated_ngrams)
print(num_updated_ngrams)

print(sets[0].issubset((sets[1])))