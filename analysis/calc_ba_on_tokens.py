import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from treetransitions.toy_data import ToyData
from treetransitions.params import Params, ObjectView
from treetransitions.utils import calc_ba

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

Params.num_seqs = [2 * 10 ** 6]
Params.num_cats_list = [[2, 8, NUM_CATS]]
Params.truncate_num_cats = [NUM_CATS]
Params.truncate_list = [[1.0, 1.0]]


for param2vals in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    probes = toy_data.probes

    for num_cats in params.num_cats_list:
        probe2cat = toy_data.num_cats2probe2cat[num_cats]
        cats = set(probe2cat.values())
        # probe2act
        probe2act = {p: np.zeros(toy_data.num_vocab) for p in probes}
        for seq in toy_data.word_sequences_mat:
            assert len(seq) == 2
            p, c = seq
            vocab_id = toy_data.word2id[c]
            probe2act[p][vocab_id] += 1
        # ba
        p_acts = [probe2act[p] for p in probes]
        ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
        print('num_cats={:>2} ba={:.2f}'.format(num_cats, ba))
    print('------------------------------------------------------')