# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from supar.modules import LSTM, MLP, BertEmbedding, CharLSTM, Highway_Concat_BiLSTM, SelfAttentionEncoder, Elmo
from supar.modules.affine import Biaffine, Triaffine
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.variational_inference import (LBPSemanticDependency,
                                                 MFVISemanticDependency)
from supar.utils import Config
from supar.utils.fn import pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
from torch.nn.utils.rnn import pad_sequence


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
                 n_elmo=1024,
                 elmo_dropout=0.33,
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

        if 'elmo' in feat:
            self.elmo_embed = Elmo(dropout=elmo_dropout)
            self.n_input += n_elmo

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
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
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
                # repr gold
                edge_pred = edges
            else:
                # repr pred
                edge_pred = s_edge.argmax(-1)
            # else:
            #     edge_pred = s_edge.ge(0).long()
            # [batch_size, seq_len]
            mask[:, 0] = 0
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
        # if(not self.args.use_pred):
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
        #     # with predicted edges
        #     edge_pred = s_edge.argmax(-1)
        #     mask1 = edge_pred.gt(0) & mask
        #     need_change_mask = mask1 & labels.eq(-1)
        #     labels = labels.masked_fill(need_change_mask, self.args.n_labels - 1)
        #     # edge loss still use gold
        #     # edge_mask = edges.gt(0) & mask
        #     edge_loss = self.criterion(s_edge[mask], edges[mask])
        #     # label loss use predicted edges
        #     if(mask1.any()):
        #         label_loss = self.criterion(s_label[mask1], labels[mask1])
        #         return self.interpolation * label_loss + (
        #             1 - self.interpolation) * edge_loss
        #     else:
        #         return edge_loss

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


    def viterbi_decode4(self, s_edge, s_label, strans, trans, mask, mask2, B_idxs, I_idxs, prd_idx):
        # 一阶段结果只提供先验概率，不作为强制条件
        edge_preds = s_edge.argmax(-1)
        pred_mask = edge_preds[..., 0].eq(1) & mask  # 所有谓词，不管有没有冲突
        label_preds = -torch.ones_like(edge_preds).long()
        label_preds[:, :, 0] = edge_preds[:, :, 0]
        label_preds.masked_fill_(label_preds.lt(1), -1)
        label_preds.masked_fill_(label_preds.eq(1), prd_idx)

    
        raw_label_num = s_label.shape[-1]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]

        # [batch_size, seq_len]
        # pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        k = pred_mask.sum()
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
        weight2 = weight2 / 2   # average the prob to O1 and O2
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

        new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        n_k = new_pred_mask.sum()  # num of the conflict predicate
        if(n_k > 0):
            pdb.set_trace()
            new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)

        return edge_preds, label_preds


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

        # [batch_size, seq_len, seq_len]
        # p_O = p_label.max(-1)[0]
        # p_O = p_label.topk(2, -1)[0][..., -1]
        # weight2 = weight2 * (p_O.unsqueeze(-1).expand(-1, -1, -1, 2))

        # weight2 = weight2 * weight2
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

        new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        n_k = new_pred_mask.sum()  # num of the conflict predicate
        if(n_k > 0):
            pdb.set_trace()
            new_pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)

        return edge_preds, label_preds



    def viterbi_decode(self, s_edge, s_label, strans, trans, mask, mask2):
        edge_preds = s_edge.argmax(-1)
        label_preds = s_label.argmax(-1)
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)
        raw_label_num = s_label.shape[-1]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        # [k, seq_len, n_labels] k is the num of predicates in this batch
        pred_scores = s_label[batch_idx, :, pred_idx, :]

        pred_scores[range(pred_idx.shape[0]), pred_idx, :] = -float('inf')

        # [k, seq_len, n_labels+2]
        pred_scores = torch.cat((pred_scores, -float('inf') * torch.ones_like(pred_scores[..., :2])), -1)
        # [k, seq_len]
        exist_e_mask = edge_preds[batch_idx, :, pred_idx].bool()
        exist_e_mask1 = exist_e_mask.unsqueeze(-1).expand(-1, -1, pred_scores.shape[-1])
        notexist_e_mask = ~exist_e_mask
        pred_scores = pred_scores.masked_fill(~exist_e_mask1, -float('inf'))
        idx1, idx2 = notexist_e_mask.nonzero()[:, 0], notexist_e_mask.nonzero()[:, 1]
        pred_scores[idx1, idx2, -2:] = 0
        # [k, seq_len-1, n_labels+2] delete the beg
        pred_scores = pred_scores[:, 1:, :]
        # pdb.set_trace()
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
            preds = torch.cat((preds, -torch.ones((k, remain_len), dtype=preds.device)), -1)
        # preds: [k, seq_len_all]
        preds = preds.masked_fill(preds.ge(raw_label_num), -1)
        label_preds = label_preds.transpose(1, 2)
        # pdb.set_trace()
        label_preds = label_preds.masked_scatter(pred_mask.unsqueeze(-1).expand(-1, -1, seq_len_all), preds)
        label_preds = label_preds.transpose(1, 2)

        # pdb.set_trace()
        return edge_preds, label_preds

    def viterbi_decode2(self, s_edge, s_label, strans, trans, mask, mask2):
        edge_preds = s_edge.argmax(-1)
        raw_seq_len = edge_preds.shape[1]
        # 直接把指向自己的边去掉
        edge_preds[:, range(raw_seq_len), range(raw_seq_len)] = 0
        edge_preds[: 0] = 0
        edge_preds = (edge_preds.gt(0) & mask2).long()
        label_preds = s_label.argmax(-1)
        # 直接把没有边的设为-1
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask
        # 有用谓词，排除掉了没有孩子的谓词
        pred_mask = pred_mask & edge_preds.sum(1).gt(0)
        all_idxs = pred_mask.nonzero()
        batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
        pred_arg_edge = edge_preds[batch_idx, :, pred_idx]
        pred_arg_score = s_label[batch_idx, :, pred_idx, :]
        idx1, idx2 = pred_arg_edge.nonzero()[:, 0], pred_arg_edge.nonzero()[:, 1]
        lengths = pred_arg_edge.sum(-1).tolist()
        tensors = pred_arg_score[idx1, idx2].split(lengths, 0)
        # [k, max_len, n_labels]
        paded_scores = pad(tensors)

        emit = paded_scores.transpose(0, 1)
        max_len, k, n_tags = emit.shape
        delta = emit.new_zeros(max_len, k, n_tags)
        paths = emit.new_zeros(max_len, k, n_tags, dtype=torch.long)
        delta[0] = strans + emit[0]  # [k, n_tags]
        for i in range(1, max_len):
            scores = trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]
        
        preds = []
        for i, length in enumerate(lengths):
            prev = torch.argmax(delta[length-1, i])
            pred = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                pred.append(prev)
            preds.append(paths.new_tensor(pred).flip(0))
        # [k, max_len]
        # preds = pad_sequence(preds, True, -1)
        src = torch.cat(preds, -1)
        back_mask = (pred_mask.unsqueeze(-1).expand(-1, -1, raw_seq_len)) & (edge_preds.transpose(1, 2).gt(0))
        label_preds = label_preds.transpose(1, 2).masked_scatter(back_mask, src).transpose(1, 2)
        # pdb.set_trace()
        return edge_preds, label_preds


class VISrlModel(nn.Module):
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
                 use_pred=False,
                 split=False,
                 encoder='lstm',
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_elmo=1024,
                 elmo_dropout=0.33,
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
        if 'elmo' in feat:
            self.elmo_embed = Elmo(dropout=elmo_dropout)
            self.n_input += n_elmo
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
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
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

        edge_d = self.mlp_un_d(x)
        edge_h = self.mlp_un_h(x)
        pair_d = self.mlp_bin_d(x)
        pair_h = self.mlp_bin_h(x)
        pair_g = self.mlp_bin_g(x)
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

        return s_edge, s_sib, s_cop, s_grd, x

    def loss(self, s_edge, s_sib, s_cop, s_grd, x, edges, labels, mask, mask2, if_eval=False):
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
        if(not if_eval):
            edge_mask = edges.gt(0) & mask
            edge_loss, marginals = self.vi((s_edge, s_sib, s_cop, s_grd), mask,
                                        edge_mask.long())
            if(not self.args.split):
                label_h = self.mlp_label_h(x)
                label_d = self.mlp_label_d(x)
            else:
                # if(self.args.repr_gold):
                edge_pred = edges
                # else:
                #     edge_pred = marginals.ge(0.5).long()
                if_prd = edge_pred[..., 0].eq(1) & mask2
                label_d = self.arg_label_d(x)
                label_h = self.arg_label_h(x)
                prd_d = self.prd_label_d(x[if_prd])
                prd_h = self.prd_label_h(x[if_prd])
                if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
                label_d = label_d.masked_scatter(if_prd, prd_d)
                label_h = label_h.masked_scatter(if_prd, prd_h)
            s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)
            
            if (edge_mask.any()):
                label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
                loss = self.interpolation * label_loss + (
                    1 - self.interpolation) * edge_loss
                return loss, marginals, s_label
            else:
                return edge_loss, marginals, s_label
        else:
            marginals = self.vi((s_edge, s_sib, s_cop, s_grd), mask)
            if(not self.args.split):
                label_h = self.mlp_label_h(x)
                label_d = self.mlp_label_d(x)
            else:
                edge_pred = marginals.ge(0.5).long()
                if_prd = edge_pred[..., 0].eq(1) & mask2
                label_d = self.arg_label_d(x)
                label_h = self.arg_label_h(x)
                prd_d = self.prd_label_d(x[if_prd])
                prd_h = self.prd_label_h(x[if_prd])
                if_prd = if_prd.unsqueeze(-1).expand(-1, -1, label_d.shape[-1])
                label_d = label_d.masked_scatter(if_prd, prd_d)
                label_h = label_h.masked_scatter(if_prd, prd_h)
            s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)
            return 0, marginals, s_label

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

        pred_predicate_num = pred_mask.sum().item()

        # [batch_size, seq_len]
        pred_mask = self.detect_conflict(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
        k = pred_mask.sum().item()  # num of the conflict predicate
        if(k <= 0):
            return label_preds, pred_predicate_num, k
        
    
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

        return label_preds, pred_predicate_num, k

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

    def detect_conflict2(self, label_preds, pred_mask, B_idxs, I_idxs, prd_idx):
        """to detect whether exist conflict (now just B-I-I, not consider B_a-Ib)
            for biiii
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
                        if(seq[j] == -1 or seq[j] in B_idxs):
                            break
                        else:
                            # span_end = j
                            j += 1
            if(flag == 1):
                mask.append(True)
            else:
                mask.append(False)
        
        conflict_mask = pred_mask.new_tensor(mask)
        pred_and_conflict = pred_mask.clone()
        pred_and_conflict = pred_and_conflict.masked_scatter(pred_mask, conflict_mask)
        return pred_and_conflict

    def bii_viterbi_decode(self, s_edge, s_label, strans, trans, mask, mask2, B_idxs, I_idxs, prd_idx):
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
        pred_mask = self.detect_conflict2(label_preds, pred_mask, B_idxs, I_idxs, prd_idx)
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
        # [batch_size, seq_len, seq_len, 1]
        weight2 = p_edge[..., 0].unsqueeze(-1)

        # weight2 = weight2 / 2   # average the prob to O1 and O2
        # weight2 = weight2 * weight2

        # [batch_size, seq_len, seq_len, raw_label_num+1]
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

    def viterbi_decode2_1(self, s_edge, s_label, strans, trans, mask, mask2, I_idxs, prd_idx, pair_dict):
        # directly process edge sequence

        edge_preds = s_edge.ge(0.5).long()
        label_preds = s_label.argmax(-1)
        label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask2), -1)
        edge_mask = edge_preds.gt(0) & mask2
        
        # tmp_mask = label_preds.eq(prd_idx)
        # tmp_mask[:, :, 0] = 0
        # label_preds = label_preds.masked_fill(tmp_mask, -1)

        raw_label_num = s_label.shape[-1] # contain [prd]
        t1, seq_len_all, t2 = edge_preds.shape[0], edge_preds.shape[1], edge_preds.shape[2]
        # [batch_size, seq_len]
        pred_mask = edge_preds[..., 0].eq(1) & mask

        # [batch_size, seq_len]
        pred_mask = self.detect_conflict_edge(label_preds, pred_mask, I_idxs, prd_idx, pair_dict)
        k = pred_mask.sum()  # num of the conflict predicate
        if(k <= 0):
            return label_preds
            
        # [batch_size, seq_len, seq_len, raw_label_num]
        s_label = s_label[..., :-1]
        label_probs = s_label.softmax(-1)
        # label_probs = label_probs * s_edge.unsqueeze(-1).expand(-1,-1,-1,raw_label_num)
        # [batch_size, seq_len, seq_len]
        edge_mask[:, :, 0] = 0  # no need to consider prd edges
        # pdb.set_trace()
        # [k, seq_len, raw_label_num]
        pred_all_probs = label_probs.transpose(1,2)[pred_mask]
        # [k, seq_len]
        pred_all_edge_mask=edge_mask.transpose(1,2)[pred_mask]
        # [num_edges, raw_label_num]
        all_useful_probs=pred_all_probs[pred_all_edge_mask]
        # [k]:[11,  3,  5,  6,  3, 10,  8]
        pred_edge_num=pred_all_edge_mask.sum(1)
        edge_num_lst = pred_edge_num.tolist()
        # a list of tensor
        lst = all_useful_probs.split(edge_num_lst, dim=0)
        # [k, max_edge_num, raw_label_num]
        pred_scores = pad_sequence(lst, True).log()
        # [max_edge_num, k, raw_label_num]
        emit = pred_scores.transpose(0, 1)
        max_edge_num, batch_size, n_tags = emit.shape
        delta = emit.new_zeros(max_edge_num, batch_size, n_tags)
        paths = emit.new_zeros(max_edge_num, batch_size, n_tags, dtype=torch.long)
        # pdb.set_trace()
        delta[0] = strans + emit[0]  # [batch_size, n_tags]
        for i in range(1, max_edge_num):
            scores = trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]

        preds = []
        for i, length in enumerate(edge_num_lst):
            prev = torch.argmax(delta[length-1, i])
            pred = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                pred.append(prev)
            preds.append(paths.new_tensor(pred).flip(0))

        all_edge_labels = torch.cat(preds, -1)
        # [k, seq_len]
        src = pred_all_edge_mask.long().masked_fill(~pred_all_edge_mask, -1)
        src = src.masked_scatter(pred_all_edge_mask, all_edge_labels)
        label_preds = label_preds.transpose(1, 2)
        label_preds[pred_mask] = src
        label_preds = label_preds.transpose(1, 2)

        return label_preds

    def detect_conflict_edge(self, label_preds, pred_mask, I_idxs, prd_idx, pair_dict):
            """to detect whether exist conflict (now just B-I-I, also consider B_a-Ib)
            Args:
                label_preds ([type]): [batch_size, seq_len, seq_len]
                pred_mask ([type]): [batch_size, seq_len]
            return:
                a mask: [batch_size, seq_len] True: exist conflict
            """
            all_idxs = pred_mask.nonzero()
            batch_idx, pred_idx = all_idxs[:, 0], all_idxs[:, 1]
            # [k, seq_len]
            k_seq = label_preds[batch_idx, :, pred_idx]
            k_seq = k_seq.masked_fill(k_seq.eq(prd_idx), -1)
            lst = k_seq.tolist()
            k = k_seq.shape[0]
            for i in range(k):
                tmp = [t for t in lst[i] if t>-1]
                lst[i] = tmp
            mask = []
            for i in range(k):
                seq = lst[i]
                length = len(seq)
                if(length == 0):
                    mask.append(False)
                    continue
                
                pre_idx = -1
                cur_idx = seq[0]

                flag = 0
                for j in range(length):
                    cur_idx = seq[j]
                    if((cur_idx in I_idxs) and (pre_idx in I_idxs)):
                        flag = 1
                        break
                    elif((cur_idx in I_idxs) and (pair_dict[cur_idx] != pre_idx)):
                        flag = 1
                        break
                    else:
                        pre_idx = seq[j]
                if(flag == 0):
                    mask.append(False)
                else:
                    mask.append(True)
                
            conflict_mask = pred_mask.new_tensor(mask)
            pred_and_conflict = pred_mask.clone()
            pred_and_conflict = pred_and_conflict.masked_scatter(pred_mask, conflict_mask)
            return pred_and_conflict