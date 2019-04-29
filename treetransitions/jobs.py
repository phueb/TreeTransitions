from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import yaml
import pandas as pd
import sys

from treetransitions.toy_data import ToyData
from treetransitions.utils import calc_ba
from treetransitions.rnn import RNN
from treetransitions.params import ObjectView
from treetransitions import config


def main_job(param2val, min_probe_freq=10):
    # check if host is down - do this before any computation
    results_p = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name'] / 'results.csv'
    assert config.RemoteDirs.runs.exists()    # this throws error if host is down

    # params
    params = ObjectView(param2val.copy())
    for k, v in param2val.items():
        print('{}={}'.format(k, v))
    print()

    toy_data = ToyData(params)

    # check probe frequency
    c = Counter(toy_data.word_sequences_mat.flatten())
    for p in toy_data.probes:
        if c[p] < min_probe_freq:
            print('WARNING: "{}" occurs only {} times'.format(p, c[p]))

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
                wx = srn.get_wx()  # TODO also test wy
                #
                probes = toy_data.probes
                probe2cat = toy_data.num_cats2probe2cat[num_cats]
                p_acts = np.asarray([wx[toy_data.word2id[p], :] for p in probes])
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

    #  save results to disk
    if not results_p.parent.exists():
        results_p.parent.mkdir(parents=True)
    traj_df = pd.DataFrame(num_cats2bas)
    with results_p.open('w') as f:
        traj_df.to_csv(f, index=False)

    # write param2val to shared drive
    param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)