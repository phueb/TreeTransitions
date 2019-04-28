import numpy as np

from treetransitions.toy_data import ToyData
from treetransitions.params import DefaultParams, ObjectView

from ludwigcluster.utils import list_all_param2vals


NUM_CATS = 32

DefaultParams.num_seqs = [2 * 10 ** 6]
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

    # TODO calc ba from tokens

    print('------------------------------------------------------')