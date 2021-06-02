import allennlp.modules.elmo
import torch


class Elmo(allennlp.modules.elmo.Elmo):
    def __init__(self, layer=3, dropout=0.5, requires_grad=False):
        """

        Args:
            layer (int):
            dropout (float):
        """
        self.n_layers = layer
        self.dropout_rate = dropout
        self.if_requires_grad = requires_grad
        super(Elmo, self).__init__(options_file="data/options.json",
                                   weight_file="data/weights.hdf5",
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
        res = super(Elmo, self).forward(chars)['elmo_representations'][0]
        return res

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout_rate > 0:
            s += f", dropout={self.dropout_rate}"
        if self.if_requires_grad:
            s += f", requires_grad={self.if_requires_grad}"

        return f"{self.__class__.__name__}({s})"
