
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input
    probe_probs = [[0.5, 0.25], [0.25, 0.5]]
    syn_cats = [['v']]  # consider making this an odd number (to speedup GPU computation)
    num_contexts = [512]
    num_seqs = [5 * 10 ** 6]
    mutation_prob = [0.01]
    num_probes = [512]  # this is also used for size of syntactic categories
    num_cats_list = [[2, 4, 8, 16, 32]]
    # rnn
    num_iterations = [10]
    num_partitions = [2]
    rnn_type = ['srn']
    mb_size = [64]
    learning_rate = [0.03]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['adagrad']  # don't forget to change learning rate
    # eval
    w = ['embeds']


for probs in Params.probe_probs:
    if probs[0] != probs[1]:
        for num_partitions in Params.num_partitions:
            assert num_partitions != 1  # no incremental structure without num_partitions > 1
