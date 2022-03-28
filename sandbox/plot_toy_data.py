
from treetransitions.toy_data import ToyData
from treetransitions.params import param2default, Params

params = Params.from_param2val(param2default)
toy_data = ToyData(params, max_ba=False, make_sequences=False)

for num_cats in toy_data.params.num_cats_list:
    toy_data.plot_tree(num_cats)

