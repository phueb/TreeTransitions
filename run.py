import argparse
import pickle
import socket
from datetime import datetime

from treetransitions import config
from treetransitions.jobs import main_job
from treetransitions.params import Params

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.Eval.plot_legal_mats = False
    config.Eval.plot_legal_corr_mats = False
    config.Eval.plot_tree = False
    #
    p = config.RemoteDirs.root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:
        main_job(param2val)
    #
    print('Finished all TreeTransitions jobs at {}.'.format(datetime.now()))
    print()


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    #
    for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
        if config.Eval.debug:
            param2val['num_seqs'] = 1 * 10 ** 3
            param2val['num_partitions'] = 2
            param2val['num_iterations'] = 1
            print('DEBUG=True: num_seqs={}'.format(param2val['num_seqs']))
        main_job(param2val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    #
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()