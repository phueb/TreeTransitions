import numpy as np

from treetransitions.params import Params, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

Params.num_seqs = [5 * 10 ** 6]
Params.num_cats_list = [[32]]
Params.legal_probs = [[1.0, 1.0], [0.5, 0.5]]

sets = []
for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print('{}={}'.format(k, v))
    print()

    # toy data
    toy_data = ToyData(params, max_ba=True)

    print(toy_data.id_sequences_mat[:10])

    # n-grams
    ngram_size = toy_data.id_sequences_mat.shape[1]
    unique_ngrams = np.unique(toy_data.id_sequences_mat, axis=0)
    num_ngrams = len(unique_ngrams)
    sets.append(set([tuple(ngram) for ngram in unique_ngrams]))
    print('num_{}-grams={:,}'.format(ngram_size, num_ngrams))
    print('------------------------------------------------------------')


updated_ngrams = sets[1].copy()
updated_ngrams.update(sets[0])
num_updated_ngrams = len(updated_ngrams)
print(num_updated_ngrams)

print('is second set of n-grams a subset of first set?')
print(sets[1].issubset((sets[0])))