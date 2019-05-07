

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input data
    truncate_type = ['probes']
    truncate_control = [False]
    truncate_num_cats = [32]
    truncate_list = [[0.5, 1.0], [1.0, 0.5]]
    num_seqs = [5 * 10 ** 6]
    num_descendants = [2]  # 2
    num_levels = [10]  # 12
    mutation_prob = [0.2]  # 0.2
    parent_count = [1024]  # exact size of single parent cluster
    num_cats_list = [[2, 4, 8, 16, 32]]
    # rnn
    num_iterations = [5]
    num_partitions = [8]
    rnn_type = ['srn']
    bptt = [1]
    mb_size = [64]
    learning_rate = [0.03]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['adagrad']
    # eval
    w = ['embeds', 'logits']