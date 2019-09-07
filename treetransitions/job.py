from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sys

from treetransitions.toy_data import ToyData
from treetransitions.utils import calc_ba
from treetransitions.rnn import RNN
from treetransitions import config


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = ''
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.fush()

    toy_data = ToyData(params, max_ba=False if config.Eval.debug else True)  # True to enable plotting of max_ba

    # train loop
    srn = RNN(toy_data.num_vocab, params)
    num_cats2bas = {num_cats: [] for num_cats in params.num_cats_list}
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
                #
                if params.w == 'embeds':
                    wx = srn.model.wx.weight.detach().cpu().numpy()
                    p_acts = np.asarray([wx[toy_data.word2id[p], :] for p in probes])
                elif params.w == 'logits':
                    x = np.asarray([[toy_data.word2id[p]] for p in probes])
                    p_acts = srn.calc_logits(x)
                else:
                    raise AttributeError('Invalid arg to "params.y".')
                ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
                num_cats2bas[num_cats].append(ba)
                print('partition={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(
                    part_num, params.num_partitions, ba, num_cats))
            print('partition={:>2}/{:>2} iteration {}/{} | before-training partition pp={:>5}\n'.format(
                part_num, params.num_partitions, iter_num, params.num_iterations, pp))
            sys.stdout.flush()

            # train
            srn.train_partition(part_id_seqs, verbose=False)  # a seq is a window (e.g. a bi-gram)

    # to pandas
    bas_df = pd.DataFrame(num_cats2bas)
    bas_df.name = 'num_cats2bas'
    max_ba_df = pd.DataFrame(toy_data.num_cats2max_ba, index=[0])  # need index because values are ints not lists
    max_ba_df.name = 'num_cats2max_ba'

    return bas_df, max_ba_df