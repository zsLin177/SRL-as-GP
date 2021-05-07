# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules.dropout import SharedDropout
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence
import torch.nn.init as init
import torch.nn.functional as F


class LSTM(nn.Module):
    r"""
    LSTM is an variant of the vanilla bidirectional LSTM adopted by Biaffine Parser
    with the only difference of the dropout strategy.
    It drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.

    APIs are roughly the same as :class:`~torch.nn.LSTM` except that we only allows
    :class:`~torch.nn.utils.rnn.PackedSequence` as input.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state `h`.
        num_layers (int):
            The number of recurrent layers. Default: 1.
        bidirectional (bool):
            If ``True``, becomes a bidirectional LSTM. Default: ``False``
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer except the last layer.
            Default: 0.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_directions = 1 + self.bidirectional

        self.f_cells = nn.ModuleList()
        if bidirectional:
            self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(
                nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(
                    nn.LSTMCell(input_size=input_size,
                                hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [
                    torch.cat((h, ih[last_batch_size:batch_size]))
                    for h, ih in zip(hx_i, hx_0)
                ]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        r"""
        Args:
            sequence (~torch.nn.utils.rnn.PackedSequence):
                A packed variable length sequence.
            hx (~torch.Tensor, ~torch.Tensor):
                A tuple composed of two tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial hidden state
                for each element in the batch.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial cell state
                for each element in the batch.
                If `hx` is not provided, both `h` and `c` default to zero.
                Default: ``None``.

        Returns:
            ~torch.nn.utils.rnn.PackedSequence, (~torch.Tensor, ~torch.Tensor):
                The first is a packed variable length sequence.
                The second is a tuple of tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the hidden state for `t=seq_len`.
                Like output, the layers can be separated using ``h.view(num_layers, num_directions, batch_size, hidden_size)``
                and similarly for c.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the cell state for `t=seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * self.num_directions, batch_size,
                             self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, self.num_directions, batch_size,
                   self.hidden_size)
        c = c.view(self.num_layers, self.num_directions, batch_size,
                   self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_i, (h_i, c_i) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            if self.bidirectional:
                x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                     hx=(h[i, 1], c[i, 1]),
                                                     cell=self.b_cells[i],
                                                     batch_sizes=batch_sizes,
                                                     reverse=True)
                x_i = torch.cat((x_i, x_b), -1)
                h_i = torch.stack((h_i, h_b))
                c_i = torch.stack((c_i, c_b))
            x = x_i
            h_n.append(h_i)
            c_n.append(h_i)

        x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal_(param)
            weight.append(param)
    return torch.cat(weight)


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class DropoutLayer(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.drop_mask = torch.Tensor(self.input_size).fill_(1 -
                                                             self.dropout_rate)
        self.drop_mask = torch.bernoulli(self.drop_mask)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def reset_dropout_mask(self, batch_size):
        self.drop_mask = torch.Tensor(
            batch_size, self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = torch.bernoulli(self.drop_mask)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def forward(self, x):
        if self.training:
            return torch.mul(x, self.drop_mask)
        else:  # eval
            return x * (1.0 - self.dropout_rate)


class MyHighwayLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyHighwayLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=6 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=5 * hidden_size,
                                   bias=False)
        self.reset_parameters()  # reset all the param in the MyLSTMCell

    def reset_parameters(self):
        weight_ih = block_orth_normal_initializer([
            self.input_size,
        ], [self.hidden_size] * 6)
        self.linear_ih.weight.data.copy_(weight_ih)

        weight_hh = block_orth_normal_initializer([
            self.hidden_size,
        ], [self.hidden_size] * 5)
        self.linear_hh.weight.data.copy_(weight_hh)
        # nn.init.constant(self.linear_hh.weight, 1.0)
        # nn.init.constant(self.linear_ih.weight, 1.0)

        nn.init.constant(self.linear_ih.bias, 0.0)

    def forward(self, x, mask=None, hx=None, dropout=None):
        assert mask is not None and hx is not None
        _h, _c = hx
        _x = self.linear_ih(x)  # compute the x
        preact = self.linear_hh(_h) + _x[:, :self.hidden_size * 5]

        i, f, o, t, j = preact.chunk(chunks=5, dim=1)
        i, f, o, t, j = F.sigmoid(i), F.sigmoid(f + 1.0), F.sigmoid(
            o), F.sigmoid(t), F.tanh(j)
        k = _x[:, self.hidden_size * 5:]

        c = f * _c + i * j
        c = mask * c + (1.0 - mask) * _c

        h = t * o * F.tanh(c) + (1.0 - t) * k
        if dropout is not None:
            h = dropout(h)
        h = mask * h + (1.0 - mask) * _h
        return h, c


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size + self.hidden_size,
                                out_features=3 * hidden_size)
        self.reset_parameters()  # reset all the param in the MyLSTMCell

    def reset_parameters(self):
        weight = block_orth_normal_initializer([
            self.input_size + self.hidden_size,
        ], [self.hidden_size] * 3)
        self.linear.weight.data.copy_(weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x, mask=None, hx=None, dropout=None):
        assert mask is not None and hx is not None
        _h, _c = hx
        _h = dropout(_h)
        _x = self.linear(torch.cat([x, _h], 1))  # compute the x
        i, j, o = _x.chunk(3, dim=1)
        i = torch.sigmoid(i)
        c = (1.0 - i) * _c + i * torch.tanh(j)
        c = mask * c  # + (1.0 - mask) * _c
        h = torch.tanh(c) * torch.sigmoid(o)
        h = mask * h  # + (1.0 - mask) * _h

        # i, f, o, t, j = preact.chunk(chunks=5, dim=1)
        # i, f, o, t, j = F.sigmoid(i), F.sigmoid(f + 1.0), F.sigmoid(o), F.sigmoid(t), F.tanh(j)
        # k = _x[:, self.hidden_size * 5:]
        #
        # c = f * _c + i * j
        # c = mask * c + (1.0 - mask) * _c

        # h = t * o * F.tanh(c) + (1.0 - t) * k
        # if dropout is not None:
        #     h = dropout(h)
        # h = mask * h + (1.0 - mask) * _h
        return h, c


class Highway_Concat_BiLSTM(nn.Module):
    """A module that runs multiple steps of HighwayBiLSTM."""
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_first=False,
                 bidirectional=False,
                 dropout_in=0,
                 dropout_out=0):
        super(Highway_Concat_BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells, self.f_dropout, self.f_hidden_dropout = [], [], []
        self.bcells, self.b_dropout, self.b_hidden_dropout = [], [], []
        self.f_initial, self.b_initial = [], []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else 2 * hidden_size if self.bidirectional else hidden_size
            self.fcells.append(
                MyLSTMCell(input_size=layer_input_size,
                           hidden_size=hidden_size))
            self.f_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.f_hidden_dropout.append(
                DropoutLayer(hidden_size, self.dropout_out))
            self.f_initial.append(
                nn.Parameter(torch.Tensor(2, self.hidden_size)))
            assert self.bidirectional is True
            self.bcells.append(
                MyLSTMCell(input_size=layer_input_size,
                           hidden_size=hidden_size))
            self.b_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.b_hidden_dropout.append(
                DropoutLayer(hidden_size, self.dropout_out))
            self.b_initial.append(
                nn.Parameter(torch.Tensor(2, self.hidden_size)))
        self.lstm_project_layer = nn.ModuleList([
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            for _ in range(num_layers - 1)
        ])
        self.fcells, self.bcells = nn.ModuleList(self.fcells), nn.ModuleList(
            self.bcells)
        self.f_dropout, self.b_dropout = nn.ModuleList(
            self.f_dropout), nn.ModuleList(self.b_dropout)
        self.f_hidden_dropout, self.b_hidden_dropout = \
            nn.ModuleList(self.f_hidden_dropout), nn.ModuleList(self.b_hidden_dropout)
        self.f_initial, self.b_initial = nn.ParameterList(
            self.f_initial), nn.ParameterList(self.b_initial)
        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"
        if self.dropout_in > 0:
            s += f", dropout={self.dropout_in}"
        if self.dropout_out > 0:
            s += f", dropout={self.dropout_out}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for layer_initial in [self.f_initial, self.b_initial]:
            for initial in layer_initial:
                init.xavier_uniform_(initial)
        for layer in self.lstm_project_layer:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)

    def reset_dropout_layer(self, batch_size):
        for layer in range(self.num_layers):
            self.f_dropout[layer].reset_dropout_mask(batch_size)
            self.f_hidden_dropout[layer].reset_dropout_mask(batch_size)
            if self.bidirectional:
                self.b_dropout[layer].reset_dropout_mask(batch_size)
                self.b_hidden_dropout[layer].reset_dropout_mask(batch_size)

    def reset_state(self, batch_size):
        f_states, b_states = [], []
        for f_layer_initial, b_layer_initial in zip(self.f_initial,
                                                    self.b_initial):
            f_states.append([
                f_layer_initial[0].expand(batch_size, -1),
                f_layer_initial[1].expand(batch_size, -1)
            ])
            b_states.append([
                b_layer_initial[0].expand(batch_size, -1),
                b_layer_initial[1].expand(batch_size, -1)
            ])
        return f_states, b_states

    @staticmethod
    def _forward_rnn(cell,
                     gate,
                     input,
                     masks,
                     initial,
                     drop_masks=None,
                     hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input[time],
                                  mask=masks[time],
                                  hx=hx,
                                  dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell,
                      gate,
                      input,
                      masks,
                      initial,
                      drop_masks=None,
                      hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input[time],
                                  mask=masks[time],
                                  hx=hx,
                                  dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(
                0, 1)  # transpose: return the transpose matrix
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()

        self.reset_dropout_layer(
            batch_size)  # reset the dropout each batch forward
        f_states, b_states = self.reset_state(batch_size)

        masks = masks.expand(
            -1, -1,
            self.hidden_size)  # expand: -1 means not expand that dimension

        h_n, c_n = [], []
        outputs = []
        for layer in range(self.num_layers):
            hidden_mask, hidden_drop = self.f_dropout[
                layer], self.f_hidden_dropout[layer]
            layer_output, (layer_h_n, layer_c_n) = \
                Highway_Concat_BiLSTM._forward_rnn(cell=self.fcells[layer],
                                                   gate=None, input=input, masks=masks, initial=f_states[layer],
                                                   drop_masks=hidden_mask, hidden_drop=hidden_drop)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
            assert self.bidirectional is True
            hidden_mask, hidden_drop = self.b_dropout[
                layer], self.b_hidden_dropout[layer]
            blayer_output, (blayer_h_n, blayer_c_n) = \
                Highway_Concat_BiLSTM._forward_brnn(cell=self.bcells[layer],
                                                    gate=None, input=input, masks=masks, initial=b_states[layer],
                                                    drop_masks=hidden_mask, hidden_drop=hidden_drop)
            h_n.append(blayer_h_n)
            c_n.append(blayer_c_n)

            output = torch.cat([layer_output, blayer_output],
                               2) if self.bidirectional else layer_output
            output = F.dropout(output, self.dropout_out, self.training)
            if layer > 0:  # Highway
                highway_gates = torch.sigmoid(
                    self.lstm_project_layer[layer - 1].forward(output))
                output = highway_gates * output + (1 - highway_gates) * input
            if self.batch_first:
                outputs.append(output.transpose(1, 0))
            else:
                outputs.append(output)
            input = output

        h_n, c_n = torch.stack(h_n, 0), torch.stack(c_n, 0)
        if self.batch_first:
            output = output.transpose(
                1, 0)  # transpose: return the transpose matrix
        return output, (h_n, c_n), outputs
