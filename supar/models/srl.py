# -*- coding: utf-8 -*-

import torch.nn as nn
from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine
from supar.structs import LBPSemanticDependency, MFVISemanticDependency
from supar.utils import Config
import torch
from torch.nn.utils.rnn import pad_sequence


class BiaffineSemanticRoleLabelingModel(Model):
    r"""

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
                 gold_p=False,
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

    def forward(self, words, feats=None, edges=None, if_prd=None):
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

        x = self.encode(words, feats, if_prd)

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

        # [batch_size, seq_len, seq_len, raw_label_num]
        p_label = s_label.softmax(-1)
        if(k <= 0):
            return edge_preds, label_preds, p_label
        
        # [batch_size, seq_len, seq_len, 2]
        p_edge = s_edge.softmax(-1)

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

        return edge_preds, label_preds, p_label

    def fix_label_cft(self, label_preds, B_idxs, I_idxs, prd_idx, pair_dict, p_label):
        '''
        solve label conflicts such as B-a and I-b
        '''
        # [batch_size, seq_len]
        pred_mask = label_preds[:, :, 0].eq(prd_idx)
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len]
        k_seq = label_preds[batch_idx, :, pred_idx]
        k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
        lst = k_seq.tolist()
        k = k_seq.shape[0]
        # [k, seq_len, raw_label_num]
        k_prob = p_label[batch_idx, :, pred_idx, :]

        for i in range(k):
            seq = lst[i][1:]
            length = len(seq)
            j = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                elif(seq[j] in I_idxs):
                    print('exists position conflicts')
                    break
                else:
                    span_start = j+1
                    b_label_idx = seq[j]
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in B_idxs):
                            break
                        else:
                            span_end = j+1
                            i_label_idx = seq[j]
                            if pair_dict[i_label_idx] != b_label_idx:
                                # happen a label conflict
                                span_start_prob = k_prob[i, span_start, b_label_idx].item()
                                span_end_prob = k_prob[i, span_end, i_label_idx].item()
                                if abs(span_start_prob - span_end_prob) < 1:
                                    lst[i][span_start] = -1
                                    lst[i][span_end] = -1
                                else:
                                    if span_start_prob > span_end_prob:
                                        lst[i][span_end] = pair_dict[b_label_idx]
                                    else:
                                        lst[i][span_start] = pair_dict[i_label_idx]
                            j += 1
                            break
        new_k_seq = torch.tensor(lst, dtype=torch.long, device=label_preds.device)
        label_preds[batch_idx, :, pred_idx] = new_k_seq
        return label_preds


class VISemanticRoleLabelingModel(BiaffineSemanticRoleLabelingModel):
    r"""
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
                 gold_p=False,
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

    def forward(self, words, feats=None, if_prd=None):
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

        x = self.encode(words, feats, if_prd)

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
            if_pred: if now is during predicting

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

        # [batch_size, seq_len, seq_len, raw_label_num]
        p_label = s_label.softmax(-1)
        if(k <= 0):
            return label_preds, p_label
        
    
        # [batch_size, seq_len, seq_len, 2]
        s_edge = s_edge.unsqueeze(-1)
        p_edge = torch.cat((1-s_edge, s_edge), -1)

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

        return label_preds, p_label

    def fix_label_cft(self, label_preds, B_idxs, I_idxs, prd_idx, pair_dict, p_label):
        '''
        solve label conflicts such as B-a and I-b
        '''
        # [batch_size, seq_len]
        pred_mask = label_preds[:, :, 0].eq(prd_idx)
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len]
        k_seq = label_preds[batch_idx, :, pred_idx]
        k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
        lst = k_seq.tolist()
        k = k_seq.shape[0]
        # [k, seq_len, raw_label_num]
        k_prob = p_label[batch_idx, :, pred_idx, :]

        for i in range(k):
            seq = lst[i][1:]
            length = len(seq)
            j = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                elif(seq[j] in I_idxs):
                    print('exists position conflicts')
                    break
                else:
                    span_start = j+1
                    b_label_idx = seq[j]
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in B_idxs):
                            break
                        else:
                            span_end = j+1
                            i_label_idx = seq[j]
                            if pair_dict[i_label_idx] != b_label_idx:
                                # happen a label conflict
                                span_start_prob = k_prob[i, span_start, b_label_idx].item()
                                span_end_prob = k_prob[i, span_end, i_label_idx].item()
                                if abs(span_start_prob - span_end_prob) < 1:
                                    lst[i][span_start] = -1
                                    lst[i][span_end] = -1
                                else:
                                    if span_start_prob > span_end_prob:
                                        lst[i][span_end] = pair_dict[b_label_idx]
                                    else:
                                        lst[i][span_start] = pair_dict[i_label_idx]
                            j += 1
                            break
        new_k_seq = torch.tensor(lst, dtype=torch.long, device=label_preds.device)
        label_preds[batch_idx, :, pred_idx] = new_k_seq
        return label_preds

    def fix_label_cft_BES(self, label_preds, B_idxs, E_idxs, S_idxs, pair_dict, prd_idx, p_label):
        '''
        solve label conflicts such as B-a and I-b
        '''
        # [batch_size, seq_len]
        pred_mask = label_preds[:, :, 0].eq(prd_idx)
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len]
        k_seq = label_preds[batch_idx, :, pred_idx]
        k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
        lst = k_seq.tolist()
        k = k_seq.shape[0]
        # [k, seq_len, raw_label_num]
        k_prob = p_label[batch_idx, :, pred_idx, :]

        for i in range(k):
            seq = lst[i][1:]
            length = len(seq)
            j = 0
            while(j < length):
                if(seq[j] == -1):
                    j += 1
                elif(seq[j] in E_idxs):
                    print('exists position conflicts')
                    break
                elif(seq[j] in S_idxs):
                    j += 1
                else:
                    span_start = j+1
                    b_label_idx = seq[j]
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in B_idxs or seq[j] in S_idxs):
                            break
                        else:
                            span_end = j+1
                            i_label_idx = seq[j]
                            if pair_dict[i_label_idx] != b_label_idx:
                                # happen a label conflict
                                span_start_prob = k_prob[i, span_start, b_label_idx].item()
                                span_end_prob = k_prob[i, span_end, i_label_idx].item()
                                if abs(span_start_prob - span_end_prob) < 1:
                                    lst[i][span_start] = -1
                                    lst[i][span_end] = -1
                                else:
                                    if span_start_prob > span_end_prob:
                                        lst[i][span_end] = pair_dict[b_label_idx]
                                    else:
                                        lst[i][span_start] = pair_dict[i_label_idx]
                            j += 1
                            break
        new_k_seq = torch.tensor(lst, dtype=torch.long, device=label_preds.device)
        label_preds[batch_idx, :, pred_idx] = new_k_seq
        return label_preds

    def detect_conflict_BES(self, label_preds, pred_mask, B_idxs, E_idxs, S_idxs, B2E_dict, prd_idx):
        '''
        to detect whether exist conflict (now just B-I-I, not consider B_a-Ib)
            for schema BES
        Args:
            label_preds ([type]): [batch_size, seq_len, seq_len]
            pred_mask ([type]): [batch_size, seq_len]
        return:
            a mask: [k] True: exist conflict
        '''
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
            while j<length:
                if seq[j] == -1:
                    j += 1
                elif seq[j] in E_idxs:
                    flag = 1
                    break
                elif seq[j] in S_idxs:
                    j += 1
                else:
                    # seq[j] in B_idxs
                    st = seq[j]
                    j += 1
                    if j>=length or seq[j] in S_idxs or seq[j] in B_idxs:
                        flag = 1
                        break
                    if seq[j] in E_idxs:
                        if seq[j] == B2E_dict[st]:
                            j += 1
                        else:
                            flag = 1
                            break
                    else:
                        # seq[j] == -1
                        while j<length and seq[j] == -1:
                            j += 1
                        if j == length:
                            flag = 1
                            break
                        elif seq[j] not in E_idxs:
                            flag = 1
                            break
                        else:
                            j += 1
            if(flag == 1):
                mask.append(True)
            else:
                mask.append(False)

        conflict_mask = pred_mask.new_tensor(mask)
        pred_and_conflict = pred_mask.clone()
        pred_and_conflict = pred_and_conflict.masked_scatter(pred_mask, conflict_mask)
        return pred_and_conflict

    def viterbi_decode_BES(self, s_edge, s_label, strans, trans, mask, mask2, B_idxs, E_idxs, S_idxs, B2E_dict, prd_idx):
        '''
        for schema BES
        '''
        edge_preds = s_edge.ge(0.5).long()
        label_preds = s_label.argmax(-1)
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)

        raw_label_num = s_label.shape[-1]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask

        pred_predicate_num = pred_mask.sum().item()
        # [batch_size, seq_len]
        pred_mask = self.detect_conflict_BES(label_preds, pred_mask, B_idxs, E_idxs, S_idxs, B2E_dict, prd_idx)
        k = pred_mask.sum().item()  # num of the conflict predicate

        # [batch_size, seq_len, seq_len, raw_label_num]
        p_label = s_label.softmax(-1)

        if(k <= 0):
            return label_preds, pred_predicate_num, k, p_label

        # [batch_size, seq_len, seq_len, 2]
        s_edge = s_edge.unsqueeze(-1)
        p_edge = torch.cat((1-s_edge, s_edge), -1)

        #[batch_size, seq_len, seq_len, raw_label_num]
        weight1 = p_edge[..., 1].unsqueeze(-1).expand(-1, -1, -1, raw_label_num)
        label_probs = weight1 * p_label
        # [batch_size, seq_len, seq_len, 2]
        weight2 = p_edge[..., 0].unsqueeze(-1).expand(-1, -1, -1, 2)

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

        return label_preds, pred_predicate_num, k, p_label

