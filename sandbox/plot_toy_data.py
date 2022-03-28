
from treetransitions.toy_data import ToyData
from treetransitions.params import param2default, Params
from treetransitions.figs import plot_heatmap

params = Params.from_param2val(param2default)
toy_data = ToyData(params)


plot_heatmap(toy_data.p_mat, [], [])

for num_cats in toy_data.params.num_cats_list:
    toy_data.plot_tree(num_cats)

