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

    # artificial data
    toy_data = ToyData(params)

    # model
    rnn = RNN(toy_data.num_vocab, params)

    # train loop
    name2col = {}
    eval_steps = []
    eval_step = 0
    for part_id, part_id_seqs in enumerate(toy_data.gen_part_id_seqs()):

        print(f'num sequences in partition={len(part_id_seqs)}')

        # perplexity
        pp = rnn.calc_seqs_pp(part_id_seqs) if config.Eval.calc_pp else np.nan

        # iterations
        for iter_id in range(params.num_iterations):

            # evaluate
            eval_steps.append(eval_step)
            for num_cats in params.num_cats_list:
                probes = toy_data.xws
                probe2cat = toy_data.num_cats2probe2cat[num_cats]

                wx = rnn.model.wx.weight.detach().cpu().numpy()
                p_acts = np.asarray([wx[toy_data.token2id[p], :] for p in probes])
                ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)

                name2col.setdefault('ba_{}'.format(num_cats), []).append(ba)

                print(f'partition={part_id + 1:>2}/{params.num_parts:>2} | ba={ba:.3f} num_cats={num_cats}')

            print(f'partition={part_id + 1:>2}/{params.num_parts:>2} iteration {iter_id + 1}/{params.num_iterations} | '
                  f'before-training partition pp={pp:>5}\n', flush=True)

            # train
            rnn.train_partition(part_id_seqs, verbose=False)
            eval_step += len(part_id_seqs) // params.batch_size

    # return performance as pandas Series
    series_list = []
    for name, col in name2col.items():
        print(f'Making pandas series with name={name} and length={len(col)}')
        s = pd.Series(col, index=eval_steps)
        s.index.name = 'step'
        s.name = name
        series_list.append(s)

    return series_list
