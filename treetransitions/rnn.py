import torch
import time
import numpy as np


class RNN:
    def __init__(self,
                 input_size,
                 params,
                 num_eval_steps=1,
                 init_range=0.01,
                 num_layers=1,
                 dropout_prob=0.0,
                 grad_clip=None):
        self.input_size = input_size
        self.params = params
        self.num_eval_steps = num_eval_steps
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.init_range = init_range
        #
        self.model = TorchRNN(self.params.rnn_type, self.num_layers,
                              self.input_size, self.params.num_hiddens, self.init_range)
        self.model.cuda()  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.params.optimization == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimization == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.learning_rate)
        else:
            raise AttributeError('Invalid arg to "optimizer"')

    def gen_batches(self, seqs):
        size = len(seqs) / self.params.mb_size
        assert size.is_integer()
        for batch in np.vsplit(seqs, size):
            yield batch

    def train_partition(self, seqs, verbose):  # seqs must be 2D array
        """
        each batch contains all windows in a sequence.
        hidden states are never saved. not across windows, and not across sequences.
        this guarantees that train updates are never based on any previous leftover information - no cheating.
        """
        start_time = time.time()
        self.model.train()
        for step, batch in enumerate(self.gen_batches(seqs)):
            self.model.batch_size = len(batch)  # dynamic batch size
            x = batch[:, :-1]
            y = batch[:, -1]

            # print(x.shape)

            # forward step
            inputs = torch.cuda.LongTensor(x.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y)
            hidden = self.model.init_hidden()  # must happen, because batch size changes from seq to seq
            logits = self.model(inputs, hidden)

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-self.params.learning_rate, p.grad.data)
            else:
                self.optimizer.step()

            # console
            if step % self.num_eval_steps == 0 and verbose:
                batch_pp = np.exp(loss.item())
                secs = time.time() - start_time
                # print(x)
                # print(y)
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in partition: {:,.0f} ".format(
                    step, batch_pp, secs))

    # ///////////////////////////////////////////////// evaluation

    def calc_seqs_pp(self, seqs):
        self.model.eval()  # protects from dropout
        loss_sum = 0
        num_batches = 0
        for batch in self.gen_batches(seqs):
            self.model.batch_size = len(batch)  # dynamic batch size
            x = batch[:, :-1]
            y = batch[:, -1]
            #
            inputs = torch.cuda.LongTensor(x.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            logits = self.model(inputs, hidden)
            #
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss_sum += self.criterion(logits, targets).item()
            num_batches += 1
        #
        res = np.exp(loss_sum / num_batches)
        return res

    def calc_logits(self, x):
        self.model.eval()  # protects from dropout
        self.model.batch_size = len(x)
        #
        inputs = torch.cuda.LongTensor(x.T)  # requires [num_steps, mb_size]
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        res = self.model(inputs, hidden).detach().cpu().numpy()
        return res

    def get_w(self, which):
        if which == 'x':
            wx = self.model.wx.weight.detach().cpu().numpy()
            print('Returning wx with shape={}'.format(wx.shape))
            return wx
        elif which == 'y':
            wy = self.model.wy.weight.detach().cpu().numpy()  # if stored on gpu
            print('Returning wy with shape={}'.format(wy.shape))
            return wy


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, hidden_size, init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = None  # is set dynamically
        self.init_range = init_range
        #
        self.wx = torch.nn.Embedding(input_size, self.hidden_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.hidden_size,
                                  out_features=input_size)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.init_range, self.init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.init_range, self.init_range)

    def init_hidden(self, verbose=False):
        if verbose:
            print('Initializing hidden weights with size [{}, {}, {}]'.format(
                self.num_layers, self.batch_size, self.hidden_size))
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            res = (torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()),
                   torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()))
        else:
            res = torch.autograd.Variable(weight.new(self.num_layers,
                                                     self.batch_size,
                                                     self.hidden_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits