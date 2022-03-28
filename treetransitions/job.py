from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from treetransitions.params import Params
from treetransitions.toy_data import ToyData
from treetransitions.utils import calc_ba
from treetransitions.rnn import RNN
from treetransitions import config


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    toy_data = ToyData(params, max_ba=False if config.Eval.debug else True)  # True to enable plotting of max_ba

    srn = RNN(toy_data.num_vocab, params)

    name2col = {}

    # train loop
    for part_id, part_id_seqs in enumerate(toy_data.gen_part_id_seqs()):
        part_num = part_id + 1

        print('num part_id_seqs in partition={}'.format(len(part_id_seqs)))
        # perplexity
        pp = srn.calc_seqs_pp(part_id_seqs) if config.Eval.calc_pp else 0

        # iterations
        for iter_id in range(params.num_iterations):
            iter_num = iter_id + 1

            # ba
            for num_cats in params.num_cats_list:
                probes = toy_data.x_words
                probe2cat = toy_data.num_cats2probe2cat[num_cats]

                wx = srn.model.wx.weight.detach().cpu().numpy()
                p_acts = np.asarray([wx[toy_data.word2id[p], :] for p in probes])
                ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)

                name2col.setdefault('ba_{}'.format(num_cats), []).append(ba)

                print('partition={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(
                    part_num, params.num_partitions, ba, num_cats))

            print('partition={:>2}/{:>2} iteration {}/{} | before-training partition pp={:>5}\n'.format(
                part_num, params.num_partitions, iter_num, params.num_iterations, pp), flush=True)

            # train
            srn.train_partition(part_id_seqs, verbose=False)  # a seq is a window (e.g. a bi-gram)

    # save max_ba
    for num_cats, max_ba in toy_data.num_cats2max_ba.items():
        name2col.setdefault('max_ba_{}'.format(num_cats), []).append(max_ba)

    # return performance as pandas Series
    series_list = []
    for name, col in name2col.items():
        print('Making pandas series with name={} and length={}'.format(name, len(col)))
        s = pd.Series(col, index=np.arange(col))
        s.name = name
        series_list.append(s)

    return series_list
