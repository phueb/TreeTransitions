

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    # truncate
    truncate_type = ['probes']
    truncate_control = [False]
    truncate_list = [[1.0, 1.0]]
    # input
    num_seqs = [1 * 10 ** 6]
    num_contexts = [1024]
    mutation_probs = [[0.01, 0.01], [0.1, 0.1]]
    stop_mutation_level = [None]  # TODO test
    template_noise = [0.1, 0.3]  # TODO test
    # probes
    min_num_cats = [32]
    num_probes = [1024]
    num_cats_list = [[32, 64, 128, 256, 512]]  # only makes sense to use bigger than min_num_cats
    # rnn
    num_iterations = [20]
    num_partitions = [1]
    rnn_type = ['srn']
    bptt = [1]
    mb_size = [64]
    learning_rate = [0.03]  # 0.03-adagrad 0.3-sgd
    num_hiddens = [128]
    optimization = ['adagrad']  # don't forget to change learning rate
    # eval
    w = ['embeds']