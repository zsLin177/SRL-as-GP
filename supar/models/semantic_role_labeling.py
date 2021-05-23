# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from supar.modules import LSTM, MLP, BertEmbedding, CharLSTM, Highway_Concat_BiLSTM, SelfAttentionEncoder
from supar.modules.affine import Biaffine, Triaffine, SmallBiaffine
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.variational_inference import (LBPSemanticDependency,
                                                 MFVISemanticDependency)
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb


class BiaffineSrlModel(nn.Module):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2018.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Edge MLP size. Default: 600.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 use_pred=False,
                 split=False,
                 encoder='lstm',
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_pretrained_embed=300,
                 n_embed_proj=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if (encoder == 'lstm'):
            self.embed_proj = nn.Linear(n_pretrained_embed, n_embed_proj)
        else:
            n_embed_proj = 100
            self.embed_proj = nn.Linear(n_pretrained_embed, n_embed_proj)

        self.n_input = n_embed + n_embed_proj
        # self.n_input = n_embed

        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        if (encoder == 'lstm'):
        # the lstm layer
            self.encoder = LSTM(input_size=self.n_input,
                                hidden_size=n_lstm_hidden,
                                num_layers=n_lstm_layers,
                                bidirectional=True,
                                dropout=lstm_dropout)
            self.encoder_dropout = SharedDropout(p=lstm_dropout)

            # self.lstm = Highway_Concat_BiLSTM(input_size=self.n_input,
            #                                   hidden_size=n_lstm_hidden,
            #                                   num_layers=n_lstm_layers,
            #                                   batch_first=True,
            #                                   bidirectional=True,
            #                                   dropout_in=0.0,
            #                                   dropout_out=0.4)

            self.mlp_edge_d = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_mlp_edge,
                                    dropout=edge_mlp_dropout,
                                    activation=False)
            self.mlp_edge_h = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_mlp_edge,
                                    dropout=edge_mlp_dropout,
                                    activation=False)
            if(not split):
                self.mlp_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.mlp_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
            else:
                self.prd_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.prd_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.arg_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.arg_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)

        else:
            self.encoder = SelfAttentionEncoder(num_encoder_layers=12,
                                                emb_size=self.n_input)
            # the MLP layers
            self.mlp_edge_d = MLP(n_in=self.n_input,
                                  n_out=n_mlp_edge,
                                  dropout=edge_mlp_dropout,
                                  activation=False)
            self.mlp_edge_h = MLP(n_in=self.n_input,
                                  n_out=n_mlp_edge,
                                  dropout=edge_mlp_dropout,
                                  activation=False)
            self.mlp_label_d = MLP(n_in=self.n_input,
                                   n_out=n_mlp_label,
                                   dropout=label_mlp_dropout,
                                   activation=False)
            self.mlp_label_h = MLP(n_in=self.n_input,
                                   n_out=n_mlp_label,
                                   dropout=label_mlp_dropout,
                                   activation=False)

        # the Biaffine layers
        # if(not sig):
        self.edge_attn = Biaffine(n_in=n_mlp_edge,
                                    n_out=2,
                                    bias_x=True,
                                    bias_y=True)
        # else:
        #     # use sigmod loss
        #     self.edge_attn = Biaffine(n_in=n_mlp_edge,
        #                             bias_x=True,
        #                             bias_y=True)
        # if(not use_pred):
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                n_out=n_labels,
                                bias_x=True,
                                bias_y=True)

        self.criterion = nn.CrossEntropyLoss()
        self.interpolation = interpolation
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
        return self

    def forward(self, words, feats, edges=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """

        _, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed = torch.cat(
                (word_embed, self.embed_proj(self.pretrained(words))), -1)
            # word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        if (self.args.encoder == 'lstm'):
        # BiLSTM
            x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.encoder_dropout(x)

            # HighwayBiLSTM
            # x, (h_n, c_n), outputs = self.lstm(embed, mask)
        else:
            x = self.encoder(embed, ~mask)

        # apply MLPs to the encoder output states
        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)

        # if(not self.args.sig):
            # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # else:
        #     # [batch_size, seq_len, seq_len]
        #     s_edge = self.edge_attn(edge_d, edge_h)
        
        if(not self.args.split):
            label_d = self.mlp_label_d(x)
            label_h = self.mlp_label_h(x)
        else:
            # if(not self.args.sig):
            # [batch_size, seq_len, seq_len]
            if(edges != None):
                # 训练时使用正确的边
                edge_pred = edges
            else:
                # 预测时使用预测的边
                edge_pred = s_edge.argmax(-1)
            # else:
            #     edge_pred = s_edge.ge(0).long()
            # [batch_size, seq_len]
            if_prd = edge_pred[..., 0].eq(1) & mask
            label_d = self.arg_label_d(x)
            label_h = self.arg_label_h(x)
            prd_d = self.prd_label_d(x[if_prd])
            prd_h = self.prd_label_h(x[if_prd])
            if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
            label_d = label_d.masked_scatter(if_prd, prd_d)
            label_h = label_h.masked_scatter(if_prd, prd_h)

        # [batch_size, seq_len, seq_len, n_labels] or [batch_size, seq_len, seq_len, n_labels+1]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_label

    def loss(self, s_edge, s_label, edges, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            edges (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard edges.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        if(not self.args.use_pred):
        # with gold edges
            # if(not self.args.sig):
            edge_mask = edges.gt(0) & mask
            edge_loss = self.criterion(s_edge[mask], edges[mask])
            if(edge_mask.any()):
                label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
                return self.interpolation * label_loss + (
                    1 - self.interpolation) * edge_loss
            else:
                return edge_loss
            # else:
            #     edge_mask = edges.gt(0) & mask
            #     target = edge_mask.long()
            #     edge_loss = F.binary_cross_entropy_with_logits(s_edge[mask],
            #                                       target[mask].float())
            #     if(edge_mask.any()):
            #         label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            #         return self.interpolation * label_loss + (
            #             1 - self.interpolation) * edge_loss
            #     else:
            #         return edge_loss
        else:
            # with predicted edges
            edge_pred = s_edge.argmax(-1)
            mask1 = edge_pred.gt(0) & mask
            need_change_mask = mask1 & labels.eq(-1)
            labels = labels.masked_fill(need_change_mask, self.args.n_labels - 1)
            # edge loss still use gold
            # edge_mask = edges.gt(0) & mask
            edge_loss = self.criterion(s_edge[mask], edges[mask])
            # label loss use predicted edges
            if(mask1.any()):
                label_loss = self.criterion(s_label[mask1], labels[mask1])
                return self.interpolation * label_loss + (
                    1 - self.interpolation) * edge_loss
            else:
                return edge_loss

    def decode(self, s_edge, s_label):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted edges and labels of shape ``[batch_size, seq_len, seq_len]``.
        """
        return s_edge.argmax(-1), s_label.argmax(-1)
        # if(not self.args.use_pred):
        #     # if(not self.args.sig):
        #     return s_edge.argmax(-1), s_label.argmax(-1)
        #     # else:
        #     #     return s_edge.ge(0).long(), s_label.argmax(-1)
        # else:
        #     return s_edge.argmax(-1), s_label[..., :-1].argmax(-1)


class VISrlModel(BiaffineSrlModel):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_un (int):
            Unary factor MLP size. Default: 600.
        n_mlp_bin (int):
            Binary factor MLP size. Default: 150.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        un_mlp_dropout (float):
            The dropout ratio of unary factor MLP layers. Default: .25.
        bin_mlp_dropout (float):
            The dropout ratio of binary factor MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: 'mfvi'.
        max_iter (int):
            Max iteration times for Variational Inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 encoder='lstm',
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_embed_proj=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_un=600,
                 n_mlp_bin=150,
                 n_mlp_label=600,
                 un_mlp_dropout=.25,
                 bin_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 inference='mfvi',
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if (encoder == 'lstm'):
            self.embed_proj = nn.Linear(n_embed, n_embed_proj)
        else:
            n_embed_proj = 100
            self.embed_proj = nn.Linear(n_embed, n_embed_proj)

        self.n_input = n_embed + n_embed_proj

        # self.n_input = n_embed
        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        if (encoder == 'lstm'):
            # the lstm layer
            self.encoder = LSTM(input_size=self.n_input,
                                hidden_size=n_lstm_hidden,
                                num_layers=n_lstm_layers,
                                bidirectional=True,
                                dropout=lstm_dropout)
            self.encoder_dropout = SharedDropout(p=lstm_dropout)
        
            # the MLP layers
            self.mlp_un_d = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_un,
                                dropout=un_mlp_dropout,
                                activation=False)
            self.mlp_un_h = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_un,
                                dropout=un_mlp_dropout,
                                activation=False)
            self.mlp_bin_d = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_bin_h = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_bin_g = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_label_d = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
            self.mlp_label_h = MLP(n_in=n_lstm_hidden * 2,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
        else:
            self.encoder = SelfAttentionEncoder(num_encoder_layers=12,
                                                emb_size=self.n_input)
            self.mlp_un_d = MLP(n_in=self.n_input,
                                n_out=n_mlp_un,
                                dropout=un_mlp_dropout,
                                activation=False)
            self.mlp_un_h = MLP(n_in=self.n_input,
                                n_out=n_mlp_un,
                                dropout=un_mlp_dropout,
                                activation=False)
            self.mlp_bin_d = MLP(n_in=self.n_input,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_bin_h = MLP(n_in=self.n_input,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_bin_g = MLP(n_in=self.n_input,
                                n_out=n_mlp_bin,
                                dropout=bin_mlp_dropout,
                                activation=False)
            self.mlp_label_d = MLP(n_in=self.n_input,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
            self.mlp_label_h = MLP(n_in=self.n_input,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)

        # the affine layers
        self.edge_attn = Biaffine(n_in=n_mlp_un, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_mlp_bin, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.vi = (MFVISemanticDependency
                   if inference == 'mfvi' else LBPSemanticDependency)(max_iter)
        self.criterion = nn.CrossEntropyLoss()
        self.interpolation = interpolation
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
        return self

    def forward(self, words, feats):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible edges of shape ``[batch_size, seq_len, seq_len]``,
                dependent-head-sibling triples of shape ``[batch_size, seq_len, seq_len, seq_len]`` and
                all possible labels on each edge of shape ``[batch_size, seq_len, seq_len, n_labels]``.
        """

        _, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed = torch.cat(
                (word_embed, self.embed_proj(self.pretrained(words))), -1)
            # word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        if (self.args.encoder == 'lstm'):
        # BiLSTM
            x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.encoder_dropout(x)

            # HighwayBiLSTM
            # x, (h_n, c_n), outputs = self.lstm(embed, mask)
        else:
            x = self.encoder(embed, ~mask)

        # apply MLPs to the BiLSTM output states

        # un_d = self.mlp_un_d(x)
        # un_h = self.mlp_un_h(x)
        # bin_d = self.mlp_bin_d(x)
        # bin_h = self.mlp_bin_h(x)
        # bin_g = self.mlp_bin_g(x)
        # label_h = self.mlp_label_h(x)
        # label_d = self.mlp_label_d(x)
        # label_h = self.mlp_label_h(x)

        # # [batch_size, seq_len, seq_len]
        # s_edge = self.edge_attn(un_d, un_h)
        # # [batch_size, seq_len, seq_len, n_labels]
        # s_sib = self.sib_attn(bin_d, bin_d, bin_h).triu_()
        # s_sib = (s_sib + s_sib.transpose(-1, -2)).permute(0, 3, 1, 2)
        # # [batch_size, seq_len, seq_len, n_labels]
        # s_cop = self.cop_attn(bin_h, bin_d, bin_h).permute(0, 3, 1, 2).triu_()
        # s_cop = s_cop + s_cop.transpose(-1, -2)
        # # [batch_size, seq_len, seq_len, n_labels]
        # s_grd = self.grd_attn(bin_g, bin_d, bin_h).permute(0, 3, 1, 2)
        # # [batch_size, seq_len, seq_len, n_labels]
        # s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        # return s_edge, s_sib, s_cop, s_grd, s_label

        edge_d = self.mlp_un_d(x)
        edge_h = self.mlp_un_h(x)
        pair_d = self.mlp_bin_d(x)
        pair_h = self.mlp_bin_h(x)
        pair_g = self.mlp_bin_g(x)
        label_h = self.mlp_label_h(x)
        label_d = self.mlp_label_d(x)

        # [batch_size, seq_len, seq_len]
        s_edge = self.edge_attn(edge_d, edge_h)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->s)
        s_sib = self.sib_attn(pair_d, pair_d, pair_h)
        s_sib = (s_sib.triu() + s_sib.triu(1).transpose(-1, -2)).permute(
            0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->c)
        s_cop = self.cop_attn(pair_h, pair_d, pair_h).permute(0, 3, 1, 2)
        s_cop = s_cop.triu() + s_cop.triu(1).transpose(-1, -2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->g)
        s_grd = self.grd_attn(pair_g, pair_d, pair_h).permute(0, 3, 1, 2)
        # ? 怎么少了?
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_sib, s_cop, s_grd, s_label

    def loss(self, s_edge, s_sib, s_cop, s_grd, s_label, edges, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_cop (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-coparent triples.
            s_grd (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-grandparent triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            edges (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard edges.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = edges.gt(0) & mask
        edge_loss, marginals = self.vi((s_edge, s_sib, s_cop, s_grd), mask,
                                       edge_mask.long())
        # print(s_label[edge_mask].shape, labels[edge_mask].shape)
        if (edge_mask.any()):
            label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            loss = self.interpolation * label_loss + (
                1 - self.interpolation) * edge_loss
            return loss, marginals
        else:
            return edge_loss, marginals

    def decode(self, s_edge, s_label):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted edges and labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        # return s_edge.argmax(-1), s_label.argmax(-1)
        return s_label.argmax(-1).masked_fill_(s_edge.lt(0.5), -1)


class BiaffineSpanSrlModel(nn.Module):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2018.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Edge MLP size. Default: 600.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 n_span_labels,
                 use_pred=False,
                 split=False,
                 encoder='lstm',
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_pretrained_embed=300,
                 n_embed_proj=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_label=600,
                 n_prd=600,
                 prd_dropout=0.33,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if (encoder == 'lstm'):
            self.embed_proj = nn.Linear(n_pretrained_embed, n_embed_proj)
        else:
            n_embed_proj = 100
            self.embed_proj = nn.Linear(n_pretrained_embed, n_embed_proj)

        self.n_input = n_embed + n_embed_proj
        # self.n_input = n_embed

        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        if (encoder == 'lstm'):
        # the lstm layer
            self.encoder = LSTM(input_size=self.n_input,
                                hidden_size=n_lstm_hidden,
                                num_layers=n_lstm_layers,
                                bidirectional=True,
                                dropout=lstm_dropout)
            self.encoder_dropout = SharedDropout(p=lstm_dropout)

            # self.lstm = Highway_Concat_BiLSTM(input_size=self.n_input,
            #                                   hidden_size=n_lstm_hidden,
            #                                   num_layers=n_lstm_layers,
            #                                   batch_first=True,
            #                                   bidirectional=True,
            #                                   dropout_in=0.0,
            #                                   dropout_out=0.4)

            self.mlp_edge_d = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_mlp_edge,
                                    dropout=edge_mlp_dropout,
                                    activation=False)
            self.mlp_edge_h = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_mlp_edge,
                                    dropout=edge_mlp_dropout,
                                    activation=False)
            self.mlp_prd = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_prd,
                                    dropout=prd_dropout,
                                    activation=False)
            self.mlp_arg = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_prd//2,
                                    dropout=prd_dropout,
                                    activation=False)

            if(not split):
                self.mlp_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.mlp_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
            else:
                self.prd_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.prd_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.arg_label_d = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)
                self.arg_label_h = MLP(n_in=n_lstm_hidden * 2,
                                        n_out=n_mlp_label,
                                        dropout=label_mlp_dropout,
                                        activation=False)

        else:
            self.encoder = SelfAttentionEncoder(num_encoder_layers=12,
                                                emb_size=self.n_input)
            # the MLP layers
            self.mlp_edge_d = MLP(n_in=self.n_input,
                                  n_out=n_mlp_edge,
                                  dropout=edge_mlp_dropout,
                                  activation=False)
            self.mlp_edge_h = MLP(n_in=self.n_input,
                                  n_out=n_mlp_edge,
                                  dropout=edge_mlp_dropout,
                                  activation=False)
            self.mlp_label_d = MLP(n_in=self.n_input,
                                   n_out=n_mlp_label,
                                   dropout=label_mlp_dropout,
                                   activation=False)
            self.mlp_label_h = MLP(n_in=self.n_input,
                                   n_out=n_mlp_label,
                                   dropout=label_mlp_dropout,
                                   activation=False)

        # the Biaffine layers
        # if(not sig):
        self.edge_attn = Biaffine(n_in=n_mlp_edge,
                                    n_out=2,
                                    bias_x=True,
                                    bias_y=True)
        # else:
        #     # use sigmod loss
        #     self.edge_attn = Biaffine(n_in=n_mlp_edge,
        #                             bias_x=True,
        #                             bias_y=True)
        # if(not use_pred):
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                n_out=n_labels,
                                bias_x=True,
                                bias_y=True)
        
        self.span_attn = SmallBiaffine(n_in=n_prd,
                                n_out=n_span_labels,
                                bias_x=True,
                                bias_y=True)

        self.criterion = nn.CrossEntropyLoss()
        self.interpolation = interpolation
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
        return self

    def forward(self, words, feats, edges=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size of indices is ``[batch_size, seq_len, fix_len]`` if feat is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """

        _, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed = torch.cat(
                (word_embed, self.embed_proj(self.pretrained(words))), -1)
            # word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        if (self.args.encoder == 'lstm'):
        # BiLSTM
            x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.encoder_dropout(x)

            # HighwayBiLSTM
            # x, (h_n, c_n), outputs = self.lstm(embed, mask)
        else:
            x = self.encoder(embed, ~mask)

        # apply MLPs to the encoder output states
        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)

        # if(not self.args.sig):
            # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # else:
        #     # [batch_size, seq_len, seq_len]
        #     s_edge = self.edge_attn(edge_d, edge_h)
        
        if(not self.args.split):
            label_d = self.mlp_label_d(x)
            label_h = self.mlp_label_h(x)
        else:
            # if(not self.args.sig):
            # [batch_size, seq_len, seq_len]
            if(edges != None):
                # 训练时使用正确的边
                edge_pred = edges
            else:
                # 预测时使用预测的边
                edge_pred = s_edge.argmax(-1)
            # else:
            #     edge_pred = s_edge.ge(0).long()
            # [batch_size, seq_len]
            if_prd = edge_pred[..., 0].eq(1) & mask
            label_d = self.arg_label_d(x)
            label_h = self.arg_label_h(x)
            prd_d = self.prd_label_d(x[if_prd])
            prd_h = self.prd_label_h(x[if_prd])
            if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
            label_d = label_d.masked_scatter(if_prd, prd_d)
            label_h = label_h.masked_scatter(if_prd, prd_h)

        # [batch_size, seq_len, seq_len, n_labels] or [batch_size, seq_len, seq_len, n_labels+1]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_label, x
    
    def span_loss(self, pad_mask, spans, encoder_out):
        """[compute span loss]

        Args:
            pad_mask ([tensor]): [batch_size, seq_len, seq_len]
            spans ([tensor]): [batch_size, seq_len, seq_len, seq_len]
            encoder_out ([tensor]): [batch_size, seq_len, 2*lstm_dim]
        """
        batch_size, seq_len, _ = pad_mask.shape
        # [batch_size, seq_len, seq_len, seq_len]
        upper_mask = torch.ones_like(spans).triu_(0).gt(0)
        span_mask = spans.gt(-1) & pad_mask.unsqueeze(1).expand(-1, seq_len, -1, -1)
        final_mask = upper_mask & span_mask
        # [1,2,3,4,...] k
        target_label = spans[final_mask]
        k = target_label.shape[0]
        # [k, 4]
        source = torch.nonzero(final_mask == 1)
        batch_idx, prd_idx, ss_idx, se_idx = source[:, 0], source[:, 1], source[:, 2], source[:, 3]
        # [k, n_prd]
        prd_repr = self.mlp_prd(encoder_out[batch_idx, prd_idx])
        # [k, n_prd//2]
        arg_start_repr = self.mlp_arg(encoder_out[batch_idx, ss_idx])
        arg_end_repr = self.mlp_arg(encoder_out[batch_idx, se_idx])
        # [k, n_prd]
        arg_repr = torch.cat((arg_start_repr+arg_end_repr, arg_start_repr-arg_end_repr), -1)
        # idx = torch.tensor([range(k)], device=pad_mask.device)
        # [k, n_span_label]
        score = self.span_attn(arg_repr, prd_repr).permute(1, 0)
        span_loss = self.criterion(score, target_label)
        return span_loss

    def loss(self, s_edge, s_label, edges, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            edges (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard edges.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        if(not self.args.use_pred):
        # with gold edges
            # if(not self.args.sig):
            edge_mask = edges.gt(0) & mask
            edge_loss = self.criterion(s_edge[mask], edges[mask])
            if(edge_mask.any()):
                label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
                return self.interpolation * label_loss + (
                    1 - self.interpolation) * edge_loss
            else:
                return edge_loss
            # else:
            #     edge_mask = edges.gt(0) & mask
            #     target = edge_mask.long()
            #     edge_loss = F.binary_cross_entropy_with_logits(s_edge[mask],
            #                                       target[mask].float())
            #     if(edge_mask.any()):
            #         label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            #         return self.interpolation * label_loss + (
            #             1 - self.interpolation) * edge_loss
            #     else:
            #         return edge_loss
        else:
            # with predicted edges
            edge_pred = s_edge.argmax(-1)
            mask1 = edge_pred.gt(0) & mask
            need_change_mask = mask1 & labels.eq(-1)
            labels = labels.masked_fill(need_change_mask, self.args.n_labels - 1)
            # edge loss still use gold
            # edge_mask = edges.gt(0) & mask
            edge_loss = self.criterion(s_edge[mask], edges[mask])
            # label loss use predicted edges
            if(mask1.any()):
                label_loss = self.criterion(s_label[mask1], labels[mask1])
                return self.interpolation * label_loss + (
                    1 - self.interpolation) * edge_loss
            else:
                return edge_loss

    def decode(self, s_edge, s_label, encoder_out, pad_mask, prd_idx, B_idx, I_idx):
        """[summary]

        Args:
            s_edge ([type]): [batch_size, seq_len, seq_len, 2]
            s_label ([type]): [batch_size, seq_len, seq_len, n_bip_labels]
            x ([type]): [batch_size, seq_len, 2*lstm_dim]
            pad_mask ([type]): [batch_size, seq_len, seq_len]

        Returns:
            spans [type]: [batch_size, seq_len, seq_len, seq_len]
        """
        batch_size, seq_len, _ = pad_mask.shape
        # [batch_size, seq_len, seq_len]
        edge_pred = s_edge.argmax(-1)
        edge_pred.masked_fill_(~pad_mask, 0)
        bip_pred = s_label.argmax(-1)
        bip_pred.masked_fill_(~pad_mask, -1)
        # 没有预测到边的地方bip也不应该有
        bip_pred.masked_fill_(edge_pred.eq(0), -1)
        # 和root相连的边强制设置为[prd]
        # [batch_size, seq_len]
        pred_mask = edge_pred[..., 0].eq(1)
        pred_mask[:, 0] = 0
        tmp_mask = pred_mask.unsqueeze(-1).expand(-1, -1, seq_len).clone()
        tmp_mask[..., 1:] = 0
        bip_pred.masked_fill_(tmp_mask, prd_idx)
        k = pred_mask.sum()
        pred_idxs = pred_mask.nonzero()
        batch_idx = pred_idxs[:, 0]
        pred_word_idx = pred_idxs[:, 1]
        # [k, seq_len]
        predicate_bi_seq = bip_pred[batch_idx, :, pred_word_idx]
        # 把[prd]去掉
        predicate_bi_seq.masked_fill_(predicate_bi_seq.eq(prd_idx), -1)

        # 没办法，暂时下面不会batch化
        b_idx = []
        s_idx = []
        e_idx = []
        lst = predicate_bi_seq.tolist()
        for i in range(k):
            seq = lst[i][1:]  # [seq_len-1]
            length = len(seq)
            j = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                else:
                    span_start = j
                    span_end = -1
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] == B_idx):
                            break
                        else:
                            span_end = j
                            j += 1
                            break
                    b_idx.append(i)
                    if(span_end != -1):
                        s_idx.append(span_start+1)
                        e_idx.append(span_end+1)
                    else:
                        s_idx.append(span_start+1)
                        e_idx.append(span_start+1)
        k_spans = -torch.ones((k, seq_len, seq_len), device=encoder_out.device).long()
        k_spans[b_idx, s_idx, e_idx] = 1
        spans = -torch.ones((batch_size, seq_len, seq_len, seq_len), device=encoder_out.device).long()
        back_mask = pred_mask.unsqueeze(-1).expand(-1, -1, seq_len).unsqueeze(-1).expand(-1, -1, -1,seq_len)
        spans = spans.masked_scatter(back_mask, k_spans)

        upper_mask = torch.ones_like(spans).triu_(0).gt(0)
        span_mask = spans.eq(1) & pad_mask.unsqueeze(1).expand(-1, seq_len, -1, -1)
        final_mask = upper_mask & span_mask
        k = final_mask.sum()  # now num of arguments
        # [k, 4]
        source = torch.nonzero(final_mask == 1)
        batch_idx, prd_idx, ss_idx, se_idx = source[:, 0], source[:, 1], source[:, 2], source[:, 3]
        prd_repr = self.mlp_prd(encoder_out[batch_idx, prd_idx])
        # [k, n_prd//2]
        arg_start_repr = self.mlp_arg(encoder_out[batch_idx, ss_idx])
        arg_end_repr = self.mlp_arg(encoder_out[batch_idx, se_idx])
        # [k, n_prd]
        arg_repr = torch.cat((arg_start_repr+arg_end_repr, arg_start_repr-arg_end_repr), -1)
        # idx = torch.tensor([range(k)], device=pad_mask.device)
        # [k, n_span_label]
        pdb.set_trace()
        score = self.span_attn(arg_repr, prd_repr).permute(1, 0)

        pred_span_label = score.argmax(-1)
        spans = spans.masked_scatter(final_mask, pred_span_label)
        return spans