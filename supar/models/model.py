# -*- coding: utf-8 -*-

import torch.nn as nn
from supar.modules import SharedDropout, TransformerEmbedding, VariationalLSTM
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        self.build()

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
        return self

    def build(self):
        if self.args.encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=self.args.n_input,
                                           hidden_size=self.args.n_lstm_hidden,
                                           num_layers=self.args.n_lstm_layers,
                                           bidirectional=True,
                                           dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
            self.args.n_hidden = self.args.n_lstm_hidden * 2
        else:
            self.encoder = TransformerEmbedding(model=self.args.bert,
                                                n_layers=self.args.n_bert_layers,
                                                pad_index=self.args.pad_index,
                                                dropout=self.args.mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self):
        raise NotImplementedError

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError
