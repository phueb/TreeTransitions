

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input
    num_contexts = [1024]
    truncate_type = ['legals']
    truncate_control = [False]
    truncate_num_cats = [32]
    truncate_list = [[1.0, 1.0]]
    num_seqs = [10 * 10 ** 6]
    # branching diffusion
    num_descendants = [2]
    num_levels = [10]
    mutation_prob = [0.01]
    stop_mutation_level = [100]  # TODO 5 results in 32 categories without any lower level differentiation
    # probes
    num_probes = [1024]  # exact size of single parent cluster
    num_cats_list = [[2, 4, 8, 16, 32]]
    # rnn
    num_iterations = [40]  # TODO
    num_partitions = [1]  # TODO
    rnn_type = ['srn']
    mb_size = [64]
    learning_rate = [0.006, 0.01, 0.03]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['adagrad']  # don't forget to change learning rate
    # eval
    w = ['embeds']