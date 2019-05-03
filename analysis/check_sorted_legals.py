import numpy as np
from collections import Counter

from treetransitions.params import DefaultParams, ObjectView
from treetransitions.toy_data import ToyData

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

DefaultParams.num_seqs = [8 * 64]
DefaultParams.num_cats_list = [[NUM_CATS]]
DefaultParams.truncate_num_cats = [NUM_CATS]
DefaultParams.truncate_list = [[0.5, 0.5], [1.0, 1.0]]


for param2vals in list_all_param2vals(DefaultParams, update_d={'param_name': 'test', 'job_name': 'test'}):

    # params
    params = ObjectView(param2vals)
    for k, v in sorted(params.__dict__.items()):
        print(k, v)

    # toy data
    toy_data = ToyData(params)
    probe2cat = toy_data.num_cats2probe2cat[NUM_CATS]
    cat2legals = toy_data.num_cats2cat2yws[NUM_CATS]
    cats = np.arange(NUM_CATS)

    all_legals = []
    for w in toy_data.vocab:
        all_legals.extend(toy_data.xw2yws[w])
    print(len(all_legals))
    legal2freq = Counter(all_legals)
    print(np.mean([f for w, f in legal2freq.items()]))

    # for cat, cat_legals in cat2legals.items():
    #     cat_legal2freq = Counter(cat_legals)
    #     print('------------------------------', cat)
    #     for k, v in sorted(cat_legal2freq.items(), key=lambda i: cat_legal2freq[i[0]])[:]:
    #         freq_in_other_cats = [l.count(k) for l in cat2legals.values()]
    #         print(k, v, legal2freq[k], np.sum(freq_in_other_cats), sorted(freq_in_other_cats))

    for w in toy_data.vocab:
        print(w, [cat2legals[c].count(w) if cat2legals[c].count(w) > 27 else 0 for c in cats])


    print('------------------------------------------------------------')