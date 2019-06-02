from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml
import pandas as pd
import sys

from treetransitions.toy_data import ToyData
from treetransitions.utils import calc_ba
from treetransitions.rnn import RNN
from treetransitions.params import ObjectView
from treetransitions import config


def main_job(param2val):
    # check if host is down - do this before any computation
    assert config.RemoteDirs.runs.exists()    # this throws error if host is down

    # params
    params = ObjectView(param2val.copy())
    for k, v in param2val.items():
        print('{}={}'.format(k, v))
    print()

    toy_data = ToyData(params, max_ba=False if config.Eval.debug else True)  # True to enable plotting of max_ba

    # train loop
    srn = RNN(toy_data.num_vocab, params)
    num_cats2bas = {num_cats: [] for num_cats in params.num_cats_list}
    for part_id, part_id_seqs in enumerate(toy_data.gen_part_id_seqs()):

        print('num part_id_seqs in partition={}'.format(len(part_id_seqs)))
        # perplexity
        pp = srn.calc_seqs_pp(part_id_seqs) if config.Eval.calc_pp else 0
        # iterations
        for iteration in range(params.num_iterations):
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
                    part_id, params.num_partitions, ba, num_cats))
            #
            print('partition={:>2}/{:>2} iteration {}/{} | before-training partition pp={:>5}\n'.format(
                part_id, params.num_partitions, iteration, params.num_iterations, pp))
            sys.stdout.flush()
            # train
            srn.train_partition(part_id_seqs, verbose=False)  # a seq is a window (e.g. a bi-gram)

    #  save results to shared drive
    bas_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'num_cats2bas.csv'
    if not bas_p.parent.exists():
        bas_p.parent.mkdir(parents=True)
    bas_df = pd.DataFrame(num_cats2bas)
    with bas_p.open('w') as f:
        bas_df.to_csv(f, index=False)

    # write num_cats2max_ba to shared drive
    max_ba_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'num_cats2max_ba.csv'
    if not max_ba_p.parent.exists():
        max_ba_p.parent.mkdir(parents=True)
    max_ba_df = pd.DataFrame(toy_data.num_cats2max_ba, index=[0])  # need index because values are ints not lists
    with max_ba_p.open('w', encoding='utf8') as f:
        max_ba_df.to_csv(f, index=False)

    # write param2val to shared drive
    param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)