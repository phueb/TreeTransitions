
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # input
    num_contexts = [1024]
    truncate_type = ['legals']
    truncate_control = ['none', 'col']
    truncate_num_cats = [64]  # TODO
    truncate_list = [[0.75, 0.75]]  # TODO
    truncate_sign = [1]
    num_seqs = [10 * 10 ** 6]
    # branching diffusion
    num_descendants = [2]
    num_levels = [10]
    mutation_prob = [0.01]
    stop_mutation_level = [100]  # TODO test
    # probes
    num_probes = [1024]
    num_cats_list = [[2, 4, 8, 16, 32, 64]]
    # rnn
    num_iterations = [40]  # TODO can set this to 20 in future with num_partitions=1
    num_partitions = [1]  # TODO
    rnn_type = ['srn']
    mb_size = [64]
    learning_rate = [0.3]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['sgd']  # don't forget to change learning rate
    # eval
    w = ['embeds']


for tl in Params.truncate_list:
    if tl[0] != tl[1]:
        if not Params.num_partitions[0] > 1:
            raise RuntimeError("Setting different truncate values requires more than 1 partition.")
