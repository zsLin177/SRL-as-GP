import allennlp.modules.elmo
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Elmo(allennlp.modules.elmo.Elmo):
    def __init__(self, layer=3, dropout=0.33, if_requires_grad=False):
        """

        Args:
            layer (int):
            dropout (float):
        """
        self.scalar_mix_parameters = []
        for i in range(layer):
            tmp_lst = [-9e10, -9e10, -9e10]
            tmp_lst[i] = 1.0
            self.scalar_mix_parameters.append(tmp_lst)

        super(Elmo, self).__init__(options_file="data/ELMO/options.json",
                                   weight_file="data/ELMO/weights.hdf5",
                                   num_output_representations=layer,
                                   requires_grad=if_requires_grad,
                                   keep_sentence_boundaries=True,
                                   scalar_mix_parameters=self.scalar_mix_parameters,
                                   dropout=dropout)
        self.n_layers = layer
        self.dropout_rate = dropout
        self.if_requires_grad = if_requires_grad
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.softmax_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor([0.0])) for _ in range(layer)])
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.normal_(self.softmax.weight, 0.0, 0.01)  # softmax layer

    def forward(self, chars, word_inputs=None):
        """

        Args:
            chars:
            word_inputs:

        Returns:

        """
        normed_weights = F.softmax(torch.cat([param for param in self.softmax_weights]), dim=0)
        normed_weights = torch.split(normed_weights, 1)
        pdb.set_trace()
        res = super(Elmo, self).forward(chars)['elmo_representations']
        final = normed_weights[0] * res[0]
        for i in range(1, self.n_layers):
            final += (normed_weights[i] * res[i])
        final = self.gamma * final
        return final[:, :-1]

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout_rate > 0:
            s += f", dropout={self.dropout_rate}"
        if self.if_requires_grad:
            s += f", requires_grad={self.if_requires_grad}"

        return f"{self.__class__.__name__}({s})"


class NewElmo(allennlp.modules.elmo.Elmo):
    def __init__(self, layer=3, dropout=0.33, requires_grad=False):
        """

        Args:
            layer (int):
            dropout (float):
        """
        self.n_layers = layer
        self.dropout_rate = dropout
        self.if_requires_grad = requires_grad
        super(NewElmo, self).__init__(options_file="data/ELMO/options.json",
                                   weight_file="data/ELMO/weights.hdf5",
                                   num_output_representations=1,
                                   requires_grad=False,
                                   dropout=dropout)

    def forward(self, chars, word_inputs=None):
        """

        Args:
            chars:
            word_inputs:

        Returns:

        """
        res = super(NewElmo, self).forward(chars)['elmo_representations'][0]
        res = torch.cat((torch.zeros_like(res[:, :1, :]), res), 1)
        return res

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout_rate > 0:
            s += f", dropout={self.dropout_rate}"
        if self.if_requires_grad:
            s += f", requires_grad={self.if_requires_grad}"

        return f"{self.__class__.__name__}({s})"

