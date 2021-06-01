import allennlp.modules.elmo
import torch


class Elmo(allennlp.modules.elmo.Elmo):
    def __init__(self, layer=1, dropout=0.33, requires_grad=False):
        """

        Args:
            layer (int):
            dropout (float):
        """
        self.n_layers = layer
        self.dropout_rate = dropout
        self.if_requires_grad = requires_grad
        scalar_mix_parameters = [-9e10, -9e10, -9e10]
        scalar_mix_parameters[layer] = 1
        super(Elmo, self).__init__(options_file="data/options.json",
                                   weight_file="data/weights.hdf5",
                                   num_output_representations=1,
                                   requires_grad=False,
                                   dropout=dropout,
                                   scalar_mix_parameters=scalar_mix_parameters)

    def forward(self, chars, word_inputs=None):
        """

        Args:
            chars:
            word_inputs:

        Returns:

        """
        res = super(Elmo, self).forward(chars)['elmo_representations'][0]
        # get minus
        forward, backward = torch.chunk(res, 2, dim=-1)

        forward_minus = forward[:, 1:] - forward[:, :-1]
        forward_minus = torch.cat([forward[:, :1], forward_minus], dim=1)
        backward_minus = backward[:, :-1] - backward[:, 1:]
        backward_minus = torch.cat([backward_minus, backward[:, -1:]], dim=1)
        # [batch_size, seq_len, n_elmo]
        res = torch.cat([forward_minus, backward_minus], dim=-1)
        return res

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout_rate > 0:
            s += f", dropout={self.dropout_rate}"
        if self.if_requires_grad:
            s += f", requires_grad={self.if_requires_grad}"

        return f"{self.__class__.__name__}({s})"
