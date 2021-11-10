# -*- coding: utf-8 -*-

import pdb
from typing import no_type_check_decorator
from torch import autograd
from torch._C import dtype
import torch.nn as nn
from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine, SimpleBiaffine
from supar.structs import LBPSemanticDependency, MFVISemanticDependency
from supar.utils import Config
from supar.utils.common import MIN
import torch
from torch.nn.utils.rnn import pad_sequence


class BiaffineSemanticRoleLabelingModel(Model):
    r"""
    The implementation of Biaffine Semantic Dependency Parser (:cite:`dozat-etal-2018-simpler`).

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
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word representations. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if ``feat='char'``. Default: 100.
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
        bert_pooling (str):
            Pooling way to get token embeddings.
            Either take the first subtoken ('first'), the last subtoken ('last'), or a mean over all ('mean').
            Default: 'mean'.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        freeze (bool):
            If ``True``, freezes bert layers. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
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

    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_edge_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_edge_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        if(not split):
            self.mlp_label_d = MLP(n_in=self.args.n_hidden,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
            self.mlp_label_h = MLP(n_in=self.args.n_hidden,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
        else:
            self.prd_label_d = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.prd_label_h = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.arg_label_d = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.arg_label_h = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)

        self.edge_attn = Biaffine(n_in=n_mlp_edge,
                                  n_out=2,
                                  bias_x=True,
                                  bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None, edges=None):
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

        x = self.encode(words, feats)

        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)
        # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        if(not self.args.split):
            label_d = self.mlp_label_d(x)
            label_h = self.mlp_label_h(x)
        else:
            if(edges != None):
                # repr gold
                edge_pred = edges
            else:
                # repr pred
                edge_pred = s_edge.argmax(-1)
                if_prd = edge_pred[..., 0].eq(1)
                if_prd[:, 0] = 0
                label_d = self.arg_label_d(x)
                label_h = self.arg_label_h(x)
                prd_d = self.prd_label_d(x[if_prd])
                prd_h = self.prd_label_h(x[if_prd])
                if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
                label_d = label_d.masked_scatter(if_prd, prd_d)
                label_h = label_h.masked_scatter(if_prd, prd_h)

        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_label

    def loss(self, s_edge, s_label, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
        if (edge_mask.any()):
            label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
            return self.args.interpolation * label_loss + (
                1 - self.args.interpolation) * edge_loss
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
            ~torch.BoolTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1)
    
    def detect_conflict(self, label_preds, pred_mask, B_idxs, I_idxs, prd_idx):
        """to detect whether exist conflict (now just B-I-I, not consider B_a-Ib)
        Args:
            label_preds ([type]): [batch_size, seq_len, seq_len]
            pred_mask ([type]): [batch_size, seq_len]
        return:
            a mask: [k] True: exist conflict
        """
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len]
        k_seq = label_preds[batch_idx, :, pred_idx]
        k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
        lst = k_seq.tolist()
        k = k_seq.shape[0]

        mask = []
        for i in range(k):
            seq = lst[i][1:]
            length = len(seq)
            j = 0
            flag = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                elif(seq[j] in I_idxs):
                    flag = 1
                    break
                else:
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in B_idxs):
                            break
                        else:
                            # span_end = j
                            j += 1
                            break
            if(flag == 1):
                mask.append(True)
            else:
                mask.append(False)
        
        conflict_mask = pred_mask.new_tensor(mask)
        pred_and_conflict = pred_mask.clone()
        pred_and_conflict = pred_and_conflict.masked_scatter(pred_mask, conflict_mask)
        return pred_and_conflict


    def viterbi_decode3(self, s_edge, s_label, strans, trans, mask, mask2, B_idxs, I_idxs, prd_idx):
        edge_preds = s_edge.argmax(-1)


        label_preds = s_label.argmax(-1)
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)
        
        # tmp_mask = label_preds.eq(prd_idx)
        # tmp_mask[:, :, 0] = 0
        # label_preds = label_preds.masked_fill(tmp_mask, -1)

        raw_label_num = s_label.shape[-1]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask

        # [batch_size, seq_len]
        pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        k = pred_mask.sum()  # num of the conflict predicate
        if(k <= 0):
            return edge_preds, label_preds
        
        # [batch_size, seq_len, seq_len, 2]
        p_edge = s_edge.softmax(-1)
        # [batch_size, seq_len, seq_len, raw_label_num]
        p_label = s_label.softmax(-1)

        #[batch_size, seq_len, seq_len, raw_label_num]
        weight1 = p_edge[..., 1].unsqueeze(-1).expand(-1, -1, -1, raw_label_num)
        label_probs = weight1 * p_label
        # [batch_size, seq_len, seq_len, 2]
        weight2 = p_edge[..., 0].unsqueeze(-1).expand(-1, -1, -1, 2)

        # weight2 = weight2 / 2   # average the prob to O1 and O2

        # weight2 = weight2 * weight2

        # [batch_size, seq_len, seq_len, raw_label_num+2]
        label_probs = torch.cat((label_probs, weight2), -1)

        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len, raw_label_num+2]
        pred_scores = label_probs[batch_idx, :, pred_idx, :].log()
        # [k, seq_len-1, n_labels+2] delete the bos
        pred_scores = pred_scores[:, 1:, :]

        emit = pred_scores.transpose(0, 1)
        seq_len, batch_size, n_tags = emit.shape
        delta = emit.new_zeros(seq_len, batch_size, n_tags)
        paths = emit.new_zeros(seq_len, batch_size, n_tags, dtype=torch.long)
        # pdb.set_trace()
        delta[0] = strans + emit[0]  # [batch_size, n_tags]

        for i in range(1, seq_len):
            scores = trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]

        preds = []
        mask1 = mask[batch_idx, :][:, 1:].t()
        for i, length in enumerate(mask1.sum(0).tolist()):
            prev = torch.argmax(delta[length-1, i])
            pred = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                pred.append(prev)
            preds.append(paths.new_tensor(pred).flip(0))
        # [k, max_len]
        # pdb.set_trace()
        preds = pad_sequence(preds, True, -1)
        # pdb.set_trace()
        preds = torch.cat((-torch.ones_like(preds[..., :1]).long(), preds), -1)
        k, remain_len = preds.shape[0], seq_len_all - preds.shape[1]
        if(remain_len > 0):
            preds = torch.cat((preds, -preds.new_ones(k, remain_len, dtype=torch.long)), -1)
        # preds: [k, seq_len_all]
        # to mask O1, O2 to -1

        preds = preds.masked_fill(preds.ge(raw_label_num), -1)
        label_preds = label_preds.transpose(1, 2)
        # pdb.set_trace()
        label_preds = label_preds.masked_scatter(pred_mask.unsqueeze(-1).expand(-1, -1, seq_len_all), preds)
        label_preds = label_preds.transpose(1, 2)

        # new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        # n_k = new_pred_mask.sum()  # num of the conflict predicate
        # if(n_k > 0):
        #     pdb.set_trace()
        #     new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)

        return edge_preds, label_preds


class VISemanticRoleLabelingModel(BiaffineSemanticRoleLabelingModel):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference (:cite:`wang-etal-2019-second`).

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
        feat (list[str]):
            Additional features to use.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if ``feat='char'``. Default: 100.
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
        bert_pooling (str):
            Pooling way to get token embeddings.
            Either take the first subtoken ('first'), the last subtoken ('last'), or a mean over all ('mean').
            Default: 'mean'.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        freeze (bool):
            If ``True``, freezes bert layers. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Unary factor MLP size. Default: 600.
        n_mlp_pair (int):
            Binary factor MLP size. Default: 150.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .25.
        pair_mlp_dropout (float):
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

    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_edge=600,
                 n_mlp_pair=150,
                 n_mlp_label=600,
                 edge_mlp_dropout=.25,
                 pair_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 inference='mfvi',
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.mlp_edge_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_edge_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_edge,
                              dropout=edge_mlp_dropout,
                              activation=False)
        self.mlp_pair_d = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        self.mlp_pair_h = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        self.mlp_pair_g = MLP(n_in=self.args.n_hidden,
                              n_out=n_mlp_pair,
                              dropout=pair_mlp_dropout,
                              activation=False)
        if(not split):
            self.mlp_label_d = MLP(n_in=self.args.n_hidden,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
            self.mlp_label_h = MLP(n_in=self.args.n_hidden,
                                n_out=n_mlp_label,
                                dropout=label_mlp_dropout,
                                activation=False)
        else:
            self.prd_label_d = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.prd_label_h = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.arg_label_d = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)
            self.arg_label_h = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_label,
                                    dropout=label_mlp_dropout,
                                    activation=False)

        self.edge_attn = Biaffine(n_in=n_mlp_edge, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_mlp_pair, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.inference = (MFVISemanticDependency if inference == 'mfvi' else
                          LBPSemanticDependency)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
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

        x = self.encode(words, feats)

        edge_d = self.mlp_edge_d(x)
        edge_h = self.mlp_edge_h(x)
        pair_d = self.mlp_pair_d(x)
        pair_h = self.mlp_pair_h(x)
        pair_g = self.mlp_pair_g(x)

        # label_h = self.mlp_label_h(x)
        # label_d = self.mlp_label_d(x)

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
        # [batch_size, seq_len, seq_len, n_labels]
        # s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        # return s_edge, s_sib, s_cop, s_grd, s_label
        return s_edge, s_sib, s_cop, s_grd, x


    def loss(self, s_edge, s_sib, s_cop, s_grd, x, labels, mask, if_pred=False):
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
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        if(if_pred):
            marginals = self.inference((s_edge, s_sib, s_cop, s_grd), mask)
            if(not self.args.split):
                label_h = self.mlp_label_h(x)
                label_d = self.mlp_label_d(x)
            else:
                edge_pred = marginals.ge(0.5).long()
                if_prd = edge_pred[..., 0].eq(1) & mask[:, :, 0]
                # if_prd = edge_pred[..., 0].eq(1)
                label_d = self.arg_label_d(x)
                label_h = self.arg_label_h(x)
                prd_d = self.prd_label_d(x[if_prd])
                prd_h = self.prd_label_h(x[if_prd])
                if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
                label_d = label_d.masked_scatter(if_prd, prd_d)
                label_h = label_h.masked_scatter(if_prd, prd_h)
            s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)
            return marginals, s_label
        else:
            edge_mask = labels.ge(0) & mask
            edge_loss, marginals = self.inference((s_edge, s_sib, s_cop, s_grd),
                                                mask, edge_mask.long())
            if(not self.args.split):
                label_h = self.mlp_label_h(x)
                label_d = self.mlp_label_d(x)
            else:
                edge_pred = marginals.ge(0.5).long()
                if_prd = edge_pred[..., 0].eq(1) & mask[:, :, 0]
                # if_prd = edge_pred[..., 0].eq(1)
                label_d = self.arg_label_d(x)
                label_h = self.arg_label_h(x)
                prd_d = self.prd_label_d(x[if_prd])
                prd_h = self.prd_label_h(x[if_prd])
                if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
                label_d = label_d.masked_scatter(if_prd, prd_d)
                label_h = label_h.masked_scatter(if_prd, prd_h)
            s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

            if(edge_mask.any()):
                label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
                loss = self.args.interpolation * label_loss + (
                    1 - self.args.interpolation) * edge_loss
                return loss, marginals, s_label
            else:
                return edge_loss, marginals, s_label

    def decode(self, s_edge, s_label):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_edge.lt(0.5), -1)


    def detect_conflict(self, label_preds, pred_mask, B_idxs, I_idxs, prd_idx):
        """to detect whether exist conflict (now just B-I-I, not consider B_a-Ib)
        Args:
            label_preds ([type]): [batch_size, seq_len, seq_len]
            pred_mask ([type]): [batch_size, seq_len]
        return:
            a mask: [k] True: exist conflict
        """
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len]
        k_seq = label_preds[batch_idx, :, pred_idx]
        k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
        lst = k_seq.tolist()
        k = k_seq.shape[0]

        mask = []
        for i in range(k):
            seq = lst[i][1:]
            length = len(seq)
            j = 0
            flag = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                elif(seq[j] in I_idxs):
                    flag = 1
                    break
                else:
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in B_idxs):
                            break
                        else:
                            # span_end = j
                            j += 1
                            break
            if(flag == 1):
                mask.append(True)
            else:
                mask.append(False)
        
        conflict_mask = pred_mask.new_tensor(mask)
        pred_and_conflict = pred_mask.clone()
        pred_and_conflict = pred_and_conflict.masked_scatter(pred_mask, conflict_mask)
        return pred_and_conflict

    def viterbi_decode3(self, s_edge, s_label, strans, trans, mask, mask2, B_idxs, I_idxs, prd_idx):
        edge_preds = s_edge.ge(0.5).long()


        label_preds = s_label.argmax(-1)
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)
        
        # tmp_mask = label_preds.eq(prd_idx)
        # tmp_mask[:, :, 0] = 0
        # label_preds = label_preds.masked_fill(tmp_mask, -1)

        raw_label_num = s_label.shape[-1]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask

        # [batch_size, seq_len]
        pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        k = pred_mask.sum()  # num of the conflict predicate
        if(k <= 0):
            return label_preds
        
    
        # [batch_size, seq_len, seq_len, 2]
        s_edge = s_edge.unsqueeze(-1)
        p_edge = torch.cat((1-s_edge, s_edge), -1)
        # [batch_size, seq_len, seq_len, raw_label_num]
        p_label = s_label.softmax(-1)

        #[batch_size, seq_len, seq_len, raw_label_num]
        weight1 = p_edge[..., 1].unsqueeze(-1).expand(-1, -1, -1, raw_label_num)
        label_probs = weight1 * p_label
        # [batch_size, seq_len, seq_len, 2]
        weight2 = p_edge[..., 0].unsqueeze(-1).expand(-1, -1, -1, 2)
        # weight2 = weight2 / 2   # average the prob to O1 and O2
        # [batch_size, seq_len, seq_len, raw_label_num+2]
        label_probs = torch.cat((label_probs, weight2), -1)

        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len, raw_label_num+2]
        pred_scores = label_probs[batch_idx, :, pred_idx, :].log()
        # [k, seq_len-1, n_labels+2] delete the bos
        pred_scores = pred_scores[:, 1:, :]

        emit = pred_scores.transpose(0, 1)
        seq_len, batch_size, n_tags = emit.shape
        delta = emit.new_zeros(seq_len, batch_size, n_tags)
        paths = emit.new_zeros(seq_len, batch_size, n_tags, dtype=torch.long)
        # pdb.set_trace()
        delta[0] = strans + emit[0]  # [batch_size, n_tags]

        for i in range(1, seq_len):
            scores = trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]

        preds = []
        mask1 = mask[batch_idx, :][:, 1:].t()
        for i, length in enumerate(mask1.sum(0).tolist()):
            prev = torch.argmax(delta[length-1, i])
            pred = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                pred.append(prev)
            preds.append(paths.new_tensor(pred).flip(0))
        # [k, max_len]
        # pdb.set_trace()
        preds = pad_sequence(preds, True, -1)
        # pdb.set_trace()
        preds = torch.cat((-torch.ones_like(preds[..., :1]).long(), preds), -1)
        k, remain_len = preds.shape[0], seq_len_all - preds.shape[1]
        if(remain_len > 0):
            preds = torch.cat((preds, -preds.new_ones(k, remain_len, dtype=torch.long)), -1)
        # preds: [k, seq_len_all]
        # to mask O1, O2 to -1

        preds = preds.masked_fill(preds.ge(raw_label_num), -1)
        label_preds = label_preds.transpose(1, 2)
        # pdb.set_trace()
        label_preds = label_preds.masked_scatter(pred_mask.unsqueeze(-1).expand(-1, -1, seq_len_all), preds)
        label_preds = label_preds.transpose(1, 2)

        # new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        # n_k = new_pred_mask.sum()  # num of the conflict predicate
        # if(n_k > 0):
        #     pdb.set_trace()
        #     new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)

        return label_preds

class GLISemanticRoleLabelingModel(Model):
    r"""
    TODO: introduction
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 gnn=None,
                 n_gnn_layers=0,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp_predicate=300,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.25,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.gnn = gnn
        self.n_gnn_layers = n_gnn_layers
        self.n_mlp_predicate = n_mlp_predicate
        self.n_mlp_argument = n_mlp_argument
        self.n_mlp_relation = n_mlp_relation

        self.predicate_repr_mlp = MLP(n_in=self.args.n_hidden, 
                                    n_out=n_mlp_predicate,
                                    dropout=repr_mlp_dropout)
        self.arg_word_repr_mlp = MLP(n_in=self.args.n_hidden,
                                    n_out=n_mlp_argument,
                                    dropout=repr_mlp_dropout)
        self.predicate_scorer = nn.Sequential(
                                            MLP(n_in=n_mlp_predicate,
                                                n_out=n_mlp_predicate//2,
                                                dropout=scorer_mlp_dropout),
                                            MLP(n_in=n_mlp_predicate//2,
                                                n_out=1,
                                                activation=False)
                                            )
        self.argument_scorer = nn.Sequential(
                                            MLP(n_in=n_mlp_argument,
                                                n_out=n_mlp_argument//2,
                                                dropout=scorer_mlp_dropout),
                                            MLP(n_in=n_mlp_argument//2,
                                                n_out=1,
                                                activation=False)
                                            )
        self.relation_init_mlp = MLP(n_in=n_mlp_argument+n_mlp_predicate,
                                    n_out=n_mlp_relation,
                                    dropout=repr_mlp_dropout)
        self.relation_scorer = MLP(n_in=n_mlp_relation,
                                    n_out=n_labels,
                                    activation=False)
        if(gnn == 'lambda' and n_gnn_layers > 0):
            self.q_mlp = MLP(n_in=n_mlp_relation, n_out=n_mlp_relation, activation=False)
            self.k_mlp = MLP(n_in=n_mlp_relation, n_out=n_mlp_relation, activation=False)
            self.v_mlp = MLP(n_in=n_mlp_relation, n_out=n_mlp_relation, activation=False)
        elif(gnn == 'gan' and n_gnn_layers > 0):
            # now every layer use the same parameter
            self.att_W = nn.Parameter(torch.empty(n_mlp_relation, n_mlp_relation))
            self.att_B = nn.Parameter(torch.empty(n_mlp_relation, n_mlp_relation))
            nn.init.normal_(self.att_W)
            nn.init.normal_(self.att_B)
            # now use just one Biaffine
            self.bia_att = SimpleBiaffine(n_in=n_mlp_relation)
            # self.same_prd_att = Biaffine(n_in=n_mlp_relation)
            # self.diff_prd_att = Biaffine(n_in=n_mlp_relation)

            # use leakyrelu rate0.1
            self.activate = nn.LeakyReLU(0.1)


        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words, span_mask, feats=None):
        # [batch_size, 2+seq_len, n_hidden]
        x = self.encode(words, feats)
        # [batch_size, 2+seq_len, n_mlp_predicate]
        predicate_repr = self.predicate_repr_mlp(x)
        # [batch_size, 2+seq_len, n_mlp_argument]
        arg_word_repr = self.arg_word_repr_mlp(x)
        # (t0, t1, t2, t3) ti:[batch_size, 2+seq_len, n_mlp_argument//4]
        split_tup = torch.split(arg_word_repr, self.n_mlp_argument//4, -1)
        # seq_all_len = 2+seq_len
        batch_size, seq_all_len, _ = split_tup[0].shape
        tmp_lst = []
        for i in range(0, 4, 2):
            expanded0 = split_tup[i].unsqueeze(2).expand(-1, -1, seq_all_len, -1)
            expanded1 = split_tup[i+1].unsqueeze(1).expand(-1, seq_all_len, -1, -1)
            tmp_lst.append(expanded0)
            tmp_lst.append(expanded1)
        # [batch_size, 2+seq_len, 2+seq_len, n_mlp_argument]
        argument_repr = torch.cat(tmp_lst, -1)
        # [real_num, n_mlp_argument] real_num = delete padded spans
        masked_arg_repr = argument_repr[span_mask]
        # [batch_size, 2+seq_len]
        predicate_score = self.predicate_scorer(predicate_repr).squeeze(-1)
        # [real_num]
        argument_score = self.argument_scorer(masked_arg_repr).squeeze(-1)

        # # [real_num, 2+seq_len, n_mlp_relation]
        # init_rela_repr = self.relation_init_mlp(torch.cat(argument_repr.unsqueeze(3).expand(-1,-1,-1,seq_all_len,-1), predicate_repr.reshape((batch_size, 1, 1, seq_all_len, -1)).expand(-1, seq_all_len,seq_all_len,-1,-1), -1)[span_mask])

        # [batch_size, 2+seq_len, 2+seq_len, 2+seq_len, n_mlp_argument+n_mlp_predicate] [batch_size, predicate, head, tail, d]
        # catted_repr = torch.cat((argument_repr.unsqueeze(3).expand(-1,-1,-1,seq_all_len,-1), predicate_repr.reshape((batch_size, 1, 1, seq_all_len, -1)).expand(-1, seq_all_len,seq_all_len,-1,-1)), -1).permute(0, 3, 1, 2, 4)

        return predicate_score, argument_score, predicate_repr, argument_repr

    def init_rela_repr(self, p_a_mask, pred_repr, arg_repr):
        """
        p_a_mask: [batch_size, 2+seq_len, 2+seq_len, 2+seq_len]: [b, predicate, head, tail]
        pred_repr: [batch_size, 2+seq_len, d]
        arg_repr: [batch_size, 2+seq_len, 2+seq_len, d]
        """
        # [k, 4]
        all_idxs = p_a_mask.nonzero()
        # [k, d]
        needed_p_repr = pred_repr[all_idxs[:,0], all_idxs[:,1]]
        # [k, d]
        needed_a_repr = arg_repr[all_idxs[:,0], all_idxs[:,2], all_idxs[:,3]]
        # [k, n_mlp_relation]
        init_repr = self.relation_init_mlp(torch.cat((needed_p_repr, needed_a_repr), -1))
        return init_repr
    
    def filter_p_a_mask(self, p_a_mask, pad_p_mask, pad_span_mask):
        max_num = int(p_a_mask.shape[0] * p_a_mask.shape[1] * 2.5)
        # max_num = 10000
        sum_r_num = p_a_mask.sum().item()
        real_token_num = pad_p_mask.sum().item()
        paded_token_num = p_a_mask.shape[0] * p_a_mask.shape[1]
        s = f" sum_r_num: {sum_r_num:6} real_tok: {real_token_num:6} paded_tok: {paded_token_num:6} max_num: {max_num:6}"
        # print(s)
        if(sum_r_num > max_num):
            all_r_idx = p_a_mask.nonzero()
            # rate = max_num/sum_r_num
            # filt_mask = (torch.rand(all_r_idx.shape[0], device=p_a_mask.device) < rate)
            shuffle = torch.arange(1, sum_r_num+1, device=all_r_idx.device)
            shuffle = shuffle[torch.randperm(shuffle.size()[0])]
            filt_mask = shuffle.le(max_num)
            filt_idx = all_r_idx[filt_mask]
            p_a_mask[:] = 0
            b_idx, p_idx, h_idx, t_idx = filt_idx[:, 0], filt_idx[:, 1], filt_idx[:, 2], filt_idx[:, 3]
            p_a_mask[b_idx, p_idx, h_idx, t_idx] = 1
            p_a_mask = p_a_mask.bool()
        p_mask = p_a_mask.sum((2,3)).gt(0) & pad_p_mask
        span_mask = p_a_mask.sum(1).gt(0) & pad_span_mask
        return p_a_mask, p_mask, span_mask

    def lambda_inter(self, pred_mask, span_mask, p_a_mask, init_rela_repr):
        """
        predicted predicate and span mask:
            pred_mask:[batch_size, seq_len+2]
            span_mask:[batch_size, seq_len+2, seq_len+2]
            p_a_mask:[batch_size, seq_len+2, seq_len+2, seq_len+2]
        init_rela_repr:[k, n_mlp_relation]
        """
        batch_size, seq_len_all = pred_mask.shape
        d = init_rela_repr.shape[-1]
        real_span_num = span_mask.sum((1,2))-1 + pred_mask.sum(-1)-1
        max_span_num = max(real_span_num)

        # [k, 4]
        rela_nums = p_a_mask.sum().item()
        k_rela_idxs = p_a_mask.nonzero()

        b_id_mask = k_rela_idxs[:, 0].unsqueeze(-1) == k_rela_idxs[:, 0].unsqueeze(0)
        p_id_mask = k_rela_idxs[:, 1].unsqueeze(-1) == k_rela_idxs[:, 1].unsqueeze(0)
        h_id_mask = k_rela_idxs[:, 2].unsqueeze(-1) == k_rela_idxs[:, 2].unsqueeze(0)
        t_id_mask = k_rela_idxs[:, 3].unsqueeze(-1) == k_rela_idxs[:, 3].unsqueeze(0)
        span_id_mask = h_id_mask & t_id_mask
        neb_mask_1 = span_id_mask & (~p_id_mask) & b_id_mask
        neb_mask_2 = p_id_mask & (~span_id_mask) & b_id_mask
        simple_neb_mask = neb_mask_1 + neb_mask_2
        indices = simple_neb_mask.nonzero()[:,1]

        if(simple_neb_mask.sum() <=0):
            return init_rela_repr

        # [k,]
        neb_nums = simple_neb_mask.sum(-1)
        back_mask = simple_neb_mask.new_zeros(rela_nums, max_span_num).bool()
        for i in range(neb_nums.shape[0]):
            back_mask[i, :neb_nums[i]] = 1
        # pdb.set_trace()

        for i in range(self.n_gnn_layers):
            
            # q = pred_mask.new_zeros(batch_size, max_rela_num, d, dtype=torch.float)
            needed_k_context = MIN * init_rela_repr.new_ones(rela_nums, max_span_num, d)
            # k = MIN * needed_k_context.new_ones(batch_size, max_rela_num, max_span_num, d)
            needed_v_context = init_rela_repr.new_zeros(rela_nums, max_span_num, d)
            # v = needed_v_context.new_zeros(batch_size, max_rela_num, max_span_num, d)

            # [k, d]
            needed_q = self.q_mlp(init_rela_repr)
            needed_k = self.k_mlp(init_rela_repr)
            needed_v = self.v_mlp(init_rela_repr)

            m_neb_k_repr = needed_k[indices]
            needed_k_context[back_mask] = m_neb_k_repr
            needed_k_context = needed_k_context.softmax(1)

            m_neb_v_repr = needed_v[indices]
            needed_v_context[back_mask] = m_neb_v_repr

            c_lambda = torch.einsum('bik,biv->bkv', needed_k_context, needed_v_context)
            init_rela_repr = torch.einsum('bk,bkv->bv', needed_q, c_lambda)
        return init_rela_repr

    def gan_inter(self, pred_mask, span_mask, p_a_mask, init_rela_repr):
        """
        predicted predicate and span mask:
            pred_mask:[batch_size, seq_len+2]
            span_mask:[batch_size, seq_len+2, seq_len+2]
            p_a_mask:[batch_size, seq_len+2, seq_len+2, seq_len+2]
        init_rela_repr:[k, n_mlp_relation]
        """
        d = init_rela_repr.shape[-1]
        real_span_num = span_mask.sum((1,2))-1 + pred_mask.sum(-1)-1
        max_span_num = max(real_span_num)

        # [k, 4]
        rela_nums = p_a_mask.sum().item()
        k_rela_idxs = p_a_mask.nonzero()

        b_id_mask = k_rela_idxs[:, 0].unsqueeze(-1) == k_rela_idxs[:, 0].unsqueeze(0)
        p_id_mask = k_rela_idxs[:, 1].unsqueeze(-1) == k_rela_idxs[:, 1].unsqueeze(0)
        h_id_mask = k_rela_idxs[:, 2].unsqueeze(-1) == k_rela_idxs[:, 2].unsqueeze(0)
        t_id_mask = k_rela_idxs[:, 3].unsqueeze(-1) == k_rela_idxs[:, 3].unsqueeze(0)
        span_id_mask = h_id_mask & t_id_mask
        neb_mask_1 = span_id_mask & (~p_id_mask) & b_id_mask
        neb_mask_2 = p_id_mask & (~span_id_mask) & b_id_mask
        simple_neb_mask = neb_mask_1 + neb_mask_2
        indices = simple_neb_mask.nonzero()[:,1]

        if(simple_neb_mask.sum() <=0):
            return init_rela_repr

        # [k,]
        neb_nums = simple_neb_mask.sum(-1)
        back_mask = simple_neb_mask.new_zeros(rela_nums, max_span_num).bool()
        for i in range(neb_nums.shape[0]):
            back_mask[i, :neb_nums[i]] = 1
        for i in range(self.n_gnn_layers):
            neb = init_rela_repr.new_zeros(rela_nums, max_span_num, d)
            neb[back_mask] = init_rela_repr[indices]
            # [rela_nums, max_span_num]
            att = self.bia_att(init_rela_repr, neb)
            att = att.masked_fill(~back_mask, MIN)
            soft_att = att.softmax(-1)
            # [rela_nums, d]
            neb_info = (soft_att.unsqueeze(-1) * neb).sum(1)
            neb_info = torch.einsum('dd,db->db', self.att_W, neb_info.transpose(0, 1)).transpose(0, 1)
            # [rela_nums, d]
            self_info = torch.einsum('dd,db->db', self.att_B, init_rela_repr.transpose(0, 1)).transpose(0, 1)
            init_rela_repr = self.activate(neb_info+self_info)
        return init_rela_repr


    def decode(self, p_score, a_score, predicate_repr, argument_repr, p_mask, span_mask):
        """
        p_score:[batch_size, 2+seq_len]
        a_score:[real_num]
        catted_repr:[batch_size, 2+seq_len, 2+seq_len, 2+seq_len, n_mlp_argument+n_mlp_predicate]
        p_mask:[batch_size, 2+seq_len] delete bos, eos and padded words
        span_mask:[batch_size, 2+seq_len, 2+seq_len] delete bos, eos and padded spans
        """
        batch_size, seq_all_len = p_score.shape
        # [batch_size, 2+seq_len]
        if_predicate = p_score.ge(0) & p_mask
        
        

        # contain the top seq_len spans as candidate spans
        span_scores = MIN * torch.ones_like(span_mask, dtype=torch.float)
        span_scores.masked_scatter_(span_mask, a_score)
        # [batch_size, seq_all_len]
        indices = torch.reshape(torch.topk(span_scores.view(batch_size, seq_all_len*seq_all_len), seq_all_len, -1)[1], (batch_size*seq_all_len,))
        b_idxs = torch.reshape(indices.new_tensor([[i]*seq_all_len for i in range(batch_size)]), (batch_size*seq_all_len,))
        if_argument = span_scores.new_zeros(batch_size, seq_all_len*seq_all_len).bool()
        if_argument[b_idxs, indices] = 1
        if_argument = torch.reshape(if_argument, (batch_size, seq_all_len, seq_all_len)) & span_mask



        # if_argument = torch.zeros_like(span_mask, dtype=torch.long)
        # # use the p > 0.5 as threshold
        # masked_if_arg = a_score.ge(0).long()
        # # [batch_size, 2+seq_len, 2+seq_len]
        # if_argument = if_argument.masked_scatter(span_mask, masked_if_arg).bool()


        # [batch_size, 2+seq_len, 2+seq_len, 2+seq_len]: [b, predicate, head, tail]
        p_a_mask = if_predicate.unsqueeze(-1).unsqueeze(-1) & if_argument.unsqueeze(1)

        if(p_a_mask.sum()<=0):
            return (-1) * torch.ones_like(p_a_mask, dtype=torch.long)

        # TODO:use the gnn rela repr
        if(self.n_gnn_layers > 0):
            p_a_mask, if_predicate, if_argument = self.filter_p_a_mask(p_a_mask, p_mask, span_mask)
            # [k, n_mlp_relation] k: true predicate x true span
            init_rela_repr = self.init_rela_repr(p_a_mask, predicate_repr, argument_repr)
            if(self.gnn == 'lambda'):
                init_rela_repr = self.lambda_inter(if_predicate, if_argument, p_a_mask, init_rela_repr)
            elif(self.gnn == 'gan'):
                init_rela_repr = self.gan_inter(if_predicate, if_argument, p_a_mask, init_rela_repr)
            else:
                raise NotImplementedError
        else:
            init_rela_repr = self.init_rela_repr(p_a_mask, predicate_repr, argument_repr)


        # [k, n_labels] n_labels contain [NULL]
        # rela_score = self.relation_scorer(init_rela_repr)
        relas = self.relation_scorer(init_rela_repr).argmax(-1)

        # (self.n_labels-1) is the index of [NULL]
        res = (self.n_labels-1) * torch.ones_like(p_a_mask, dtype=torch.long)
        res = res.masked_scatter(p_a_mask, relas)
        res = res.masked_fill(res.eq(self.n_labels-1), -1)
        return res

    @torch.enable_grad()
    def dp_decode(self, p_score, a_score, predicate_repr, argument_repr, p_mask, span_mask):
        """
        p_score:[batch_size, 2+seq_len]
        a_score:[real_num]
        predicate_repr: [batch_size, 2+seq_len, n_mlp_predicate]
        argument_repr: [batch_size, 2+seq_len, 2+seq_len, n_mlp_argument]
        p_mask:[batch_size, 2+seq_len] delete bos, eos and padded words
        span_mask:[batch_size, 2+seq_len, 2+seq_len] delete bos, eos and padded spans
        """
        batch_size, seq_all_len = p_score.shape
        # [batch_size, seq_all_len, 1]
        lens = p_mask.sum(-1).unsqueeze(-1).expand(-1, seq_all_len).unsqueeze(-1)
        # [batch_size, 2+seq_len]
        if_predicate = p_score.ge(0) & p_mask
        
        # contain the top seq_len spans as candidate spans
        span_scores = MIN * torch.ones_like(span_mask, dtype=torch.float)
        span_scores.masked_scatter_(span_mask, a_score)
        # [batch_size, seq_all_len]
        indices = torch.reshape(torch.topk(span_scores.view(batch_size, seq_all_len*seq_all_len), seq_all_len, -1)[1], (batch_size*seq_all_len,))
        b_idxs = torch.reshape(indices.new_tensor([[i]*seq_all_len for i in range(batch_size)]), (batch_size*seq_all_len,))
        if_argument = span_scores.new_zeros(batch_size, seq_all_len*seq_all_len).bool()
        if_argument[b_idxs, indices] = 1
        if_argument = torch.reshape(if_argument, (batch_size, seq_all_len, seq_all_len)) & span_mask


        # if_argument = torch.zeros_like(span_mask, dtype=torch.long)
        # # [real_num]
        # masked_if_arg = a_score.ge(0).long()
        # # [batch_size, 2+seq_len, 2+seq_len]
        # if_argument = if_argument.masked_scatter(span_mask, masked_if_arg).bool()
        # # [batch_size, 1]
        

        # [batch_size, 2+seq_len, 2+seq_len, 2+seq_len]: [b, predicate, head, tail]
        p_a_mask = if_predicate.unsqueeze(-1).unsqueeze(-1) & if_argument.unsqueeze(1)

        if(p_a_mask.sum()<=0):
            return (-1) * torch.ones_like(p_a_mask, dtype=torch.long)

        if(self.n_gnn_layers > 0):
            p_a_mask, if_predicate, if_argument = self.filter_p_a_mask(p_a_mask, p_mask, span_mask)
            if_overlap = self.detect_overlap(if_argument).unsqueeze(-1)
        else:
            if_overlap = self.detect_overlap(if_argument).unsqueeze(-1)

        # decide use local or dp
        # [batch_size, 2+seq_len]
        local_mask = if_predicate & (~if_overlap)
        dp_mask = if_predicate & if_overlap

        res = (self.n_labels-1) * torch.ones_like(p_a_mask, dtype=torch.long)

        local_p_a_mask = local_mask.unsqueeze(-1).unsqueeze(-1) & if_argument.unsqueeze(1)
        if(local_p_a_mask.sum()>0):
            # use local decode
            if(self.n_gnn_layers>0):
                init_rela_repr = self.init_rela_repr(local_p_a_mask, predicate_repr, argument_repr)
                if(self.gnn == 'lambda'):
                    init_rela_repr = self.lambda_inter(local_mask, if_argument, local_p_a_mask, init_rela_repr)
                elif(self.gnn == 'gan'):
                    init_rela_repr = self.gan_inter(local_mask, if_argument, local_p_a_mask, init_rela_repr)
                else:
                    raise NotImplementedError

            else:
                init_rela_repr = self.init_rela_repr(local_p_a_mask, predicate_repr, argument_repr)
            # relas: [m]
            relas = self.relation_scorer(init_rela_repr).argmax(-1)
            res = res.masked_scatter(local_p_a_mask, relas)
        
        dp_p_a_mask = dp_mask.unsqueeze(-1).unsqueeze(-1) & if_argument.unsqueeze(1)
        if(dp_p_a_mask.sum()>0):
            # [num_prd]
            needed_len = lens[dp_mask].squeeze(-1)
            # use dp decode
        
            if(self.n_gnn_layers>0):
                # [n, n_mlp_relation]
                init_rela_repr = self.init_rela_repr(dp_p_a_mask, predicate_repr, argument_repr)
                if(self.gnn == 'lambda'):
                    init_rela_repr = self.lambda_inter(dp_mask, if_argument, dp_p_a_mask, init_rela_repr)
                elif(self.gnn == 'gan'):
                    init_rela_repr = self.gan_inter(dp_mask, if_argument, dp_p_a_mask, init_rela_repr)
                else:
                    raise NotImplementedError
            else:
                init_rela_repr = self.init_rela_repr(dp_p_a_mask, predicate_repr, argument_repr)

            # 概率>exp()>直接mlp出来的score
            # 或许问题还是出在分数的定义上，对于确实是预测出的span的分数应该是多少，以及不是预测出的span的分数 

            # [n, n_labels]
            score_all_label = self.relation_scorer(init_rela_repr).softmax(-1)
            # [n]
            relas = score_all_label.argmax(-1)
            # [num_prd, 2+seq_len, 2+seq_len]
            tmp_res = (self.n_labels-1) * score_all_label.new_ones(dp_mask.sum().item(), seq_all_len, seq_all_len, dtype=torch.long)
            tmp_res = tmp_res.masked_scatter(dp_p_a_mask[dp_mask], relas)
            
            # [n]
            # e_score = score_all_label.max(-1)[0]
            e_score = 1-score_all_label[:, -1]

            # [num_prd, 2+seq_len, 2+seq_len]
            e_score_all = 0 * e_score.new_ones(dp_mask.sum().item(), seq_all_len, seq_all_len, requires_grad=True)
            e_score_all = e_score_all.masked_scatter(dp_p_a_mask[dp_mask], e_score)

            # e_score_all = e_score_all.exp()

            num_prd = e_score_all.shape[0]
            score = e_score_all.new_zeros(num_prd, seq_all_len)
            path = -e_score_all.new_ones(num_prd, seq_all_len, dtype=torch.long)
            # init
            path[:, 1] = 1
            score[:, 1] = e_score_all[:, 1, 1]
            # [2+seq_len, num_prd]
            score.transpose_(0, 1)
            path.transpose_(0, 1)
            # [head, tail, num_prd]
            e_score_all = e_score_all.permute(1, 2, 0)
            for i in range(1, seq_all_len):
                # [i, num_prd]
                tmp_tensor = score.new_zeros(i, num_prd)
                tmp_tensor[-1] = e_score_all[1, i]
                tmp_tensor[:-1] = score[1: i] + e_score_all[2:i+1, i]
                score[i], path[i] = tmp_tensor.max(0)
                path[i] = path[i] + 1
            
            # backtrack with backward
            # [num_prd, seq_all_len]
            score.transpose_(0, 1)
            z = score[range(num_prd), needed_len].sum()
            grd, = autograd.grad(z, e_score_all)
            back_mask = grd.permute(2, 0, 1).long()

            # normal backtrack
            # back_mask = torch.zeros_like(tmp_res, dtype=torch.long)
            # path = path.transpose(0, 1).tolist()
            # for i in range(num_prd):
            #     j = needed_len[i].item()
            #     while(j>=1):
            #         if(path[i][j] == j):
            #             back_mask[i, 1, j] = 1
            #             break
            #         else:
            #             k = path[i][j]
            #             back_mask[i, k+1, j] = 1
            #             j = k
            
            tmp_res.masked_fill_(~(back_mask.bool()), self.n_labels-1)
            res[dp_mask] = tmp_res
        
        res = res.masked_fill(res.eq(self.n_labels-1), -1)
        return res


    def detect_overlap(self, if_argument):
        """
        if_argument: [batch_size, seq_len+2, seq_len+2] bool
        return if_overlap: [batch_size] True: exist overlap need dp
        """
        batch_size, seq_all, _ = if_argument.shape
        if_overlap = [False] * batch_size
        for i in range(batch_size):
            span_lst = if_argument[i].nonzero().tolist()
            if(len(span_lst)<=1):
                continue
            for j in range(1, len(span_lst)):
                if(span_lst[j][0]<=span_lst[j-1][1]):
                    if_overlap[i] = True
                    break
        return if_argument.new_tensor(if_overlap)



    def loss(self, p_score, a_score, predicate_repr, argument_repr, p_mask, span_mask, gold_p, gold_span, gold_relas, flag=0):
        batch_size, seq_all_len = p_score.shape
        p_loss = self.bce_criterion(p_score[p_mask], gold_p[p_mask].float())
        argument_loss = self.bce_criterion(a_score, gold_span[span_mask].float())
        # here currently use gold, may use predicted
        # [batch_size, 2+seq_len, 2+seq_len, 2+seq_len]: [b, predicate, head, tail]
        
        if(flag):
            p_a_mask = gold_p.bool().unsqueeze(-1).unsqueeze(-1) & gold_span.bool().unsqueeze(1)
        # elif(not flag and self.n_gnn_layers == 0):
        #     p_a_mask = gold_p.bool().unsqueeze(-1).unsqueeze(-1) & gold_span.bool().unsqueeze(1)
        else:
            # use predicted
            pred_p = p_score.ge(0) & p_mask


            # contain the top seq_len spans as candidate spans
            span_scores = MIN * torch.ones_like(span_mask, dtype=torch.float)
            span_scores.masked_scatter_(span_mask, a_score)
            # [batch_size, seq_all_len]
            indices = torch.reshape(torch.topk(span_scores.view(batch_size, seq_all_len*seq_all_len), seq_all_len, -1)[1], (batch_size*seq_all_len,))
            b_idxs = torch.reshape(indices.new_tensor([[i]*seq_all_len for i in range(batch_size)]), (batch_size*seq_all_len,))
            pred_a = span_scores.new_zeros(batch_size, seq_all_len*seq_all_len).bool()
            pred_a[b_idxs, indices] = 1
            pred_a = torch.reshape(pred_a, (batch_size, seq_all_len, seq_all_len)) & span_mask


            # pred_a = torch.zeros_like(span_mask, dtype=torch.long)
            # # [real_num]
            # masked_if_arg = a_score.ge(0).long()
            # # [batch_size, 2+seq_len, 2+seq_len]
            # pred_a = pred_a.masked_scatter(span_mask, masked_if_arg).bool()


            p_a_mask = pred_p.unsqueeze(-1).unsqueeze(-1) & pred_a.unsqueeze(1)


        if(p_a_mask.sum()<=0):
            return p_loss + argument_loss

        if(self.n_gnn_layers > 0):
            p_a_mask, pred_p, pred_a = self.filter_p_a_mask(p_a_mask, p_mask, span_mask)
            # [k, n_mlp_relation] k: true predicate x true span
            init_rela_repr = self.init_rela_repr(p_a_mask, predicate_repr, argument_repr)
            if(self.gnn == 'lambda'):
                init_rela_repr = self.lambda_inter(pred_p, pred_a, p_a_mask, init_rela_repr)
            elif(self.gnn == 'gan'):
                init_rela_repr = self.gan_inter(pred_p, pred_a, p_a_mask, init_rela_repr)
            else:
                raise NotImplementedError
        else:
            init_rela_repr = self.init_rela_repr(p_a_mask, predicate_repr, argument_repr)
        rela_loss = self.ce_criterion(self.relation_scorer(init_rela_repr), gold_relas[p_a_mask])

        return p_loss + argument_loss + rela_loss
        
    






        
