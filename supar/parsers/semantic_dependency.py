# -*- coding: utf-8 -*-

import os
import pdb
import numpy
import torch
import torch.nn as nn
from supar.models import (BiaffineSemanticDependencyModel,
                          VISemanticDependencyModel, BiaffineSRLModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding, vocab
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric, SrlMetric
from supar.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger(__name__)


class BiaffineSemanticDependencyParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 20178.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    """

    NAME = 'biaffine-semantic-dependency'
    MODEL = BiaffineSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.LABEL = self.transform.PHEAD
        # self.EDGE, self.LABEL, self.SENSE = self.transform.PHEAD

        self.transition_params = self.get_transition_params()

    def get_transition_params(self):
        label_strs = [
            self.LABEL.vocab[idx] for idx in range(len(self.LABEL.vocab))
        ] + ['O']
        num_tags = len(label_strs)
        transition_params = numpy.zeros([num_tags, num_tags],
                                        dtype=numpy.float32)
        for i, prev_label in enumerate(label_strs):
            for j, label in enumerate(label_strs):
                if i != j and label[
                        0] == 'I' and not prev_label == 'B' + label[1:]:
                    transition_params[i, j] = numpy.NINF
        return transition_params

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), ChartMetric()

        for words, *feats, edges, labels in bar:

            self.optimizer.zero_grad()
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            edge_preds, label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                   labels.masked_fill(~(edges.gt(0) & mask), -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SrlMetric()

        for words, *feats, edges, labels in loader:
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            # loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            # total_loss += loss.item()

            edge_preds, label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                   labels.masked_fill(~(edges.gt(0) & mask), -1))
        # total_loss /= len(loader)

        # return total_loss, metric
        return metric

    @torch.no_grad()
    def viterbi_decode(self, score):
        r'''
            This should only be used at test time.
            score: A [seq_len, num_tags] matrix of unary potentials.
                    seq_len is the real len
            viterbi: A [seq_len] list of integers containing the highest scoring tag
              indicies.
            viterbi_score: A float containing the score for the Viterbi sequence.
        '''
        score = numpy.array(score)
        transition_params = self.transition_params
        trellis = numpy.zeros_like(score)
        backpointers = numpy.zeros_like(score, dtype=numpy.int32)
        trellis[0] = score[0]
        for t in range(1, score.shape[0]):
            v = numpy.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + numpy.max(v, 0)
            backpointers[t] = numpy.argmax(v, 0)
        viterbi = [numpy.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = numpy.max(trellis[-1])
        return viterbi, viterbi_score

    @torch.no_grad()
    def get_label_seqs(self, s_edge, s_label, mask, lemmas):
        r'''
            get label_index seqs for a sentence
            s_egde (~torch.Tensor): ``[seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[ seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            mask :[seq_len] filter paded tokens and root
        '''
        seq_len, _, n_labels = s_label.shape
        real_len = mask.sum().item()
        # [seq_len, seq_len] tail,head
        edge_pred = s_edge.argmax(-1)
        # [seq_len] if token is a predicate
        predicate_mask = (edge_pred[:, 0] == 1) & mask
        predicate_num = predicate_mask.sum().item()
        # [seq_len, seq_len, n_labels] head, tail, n_labels
        s_label = s_label.permute(1, 0, 2)
        # [predicate_num, seq_len, n_labels]
        prd_seq_label_scores = s_label[predicate_mask]
        # [predicate_num, seq_len] whether exist edge between predicate and token
        prd_seq_mask = (edge_pred.permute(1, 0)[predicate_mask].eq(1)) & mask
        # [predicate_num, seq_len, n_labels+1]
        prd_seq_label_scores = torch.cat(
            (prd_seq_label_scores,
             torch.zeros_like(prd_seq_label_scores[..., :1])), -1)
        prd_seq_label_scores[..., -1] = float('-inf')
        prd_seq_label_scores.masked_fill_(
            (~prd_seq_mask).unsqueeze(2).expand(-1, -1, n_labels + 1),
            float('-inf'))
        end_mask = (~prd_seq_mask) & mask
        prd_seq_label_scores[end_mask, [-1] * end_mask.sum()] = 0.0
        # list [prd1_idx, prd2_idx] start from 1
        real_idx_prd = ((predicate_mask == True).nonzero()).squeeze(1).tolist()
        columns = []
        columns.append(predicate_mask.tolist()[1:real_len + 1])
        for w in prd_seq_label_scores:
            real_score = w.tolist()[1:real_len + 1]
            viterbi, viterbi_score = self.viterbi_decode(real_score)
            label_lst = []
            for idx in viterbi:
                if (idx < len(self.LABEL.vocab)):
                    label_lst.append(self.LABEL.vocab[idx])
                else:
                    label_lst.append('O')
            # pdb.set_trace()

            columns.append(label_lst)

        for i in range(1, len(columns)):
            prd_idx = real_idx_prd[i - 1]
            new_column = self.label2span(columns[i], prd_idx)
            columns[i] = new_column

        lines = []
        for i in range(real_len):
            line = []
            # print(i)
            # print(len(columns[0]))
            # print(mask)
            # print(mask.sum())
            if (columns[0][i]):
                line.append(lemmas[i])
            else:
                line.append('-')
            for j in range(1, len(columns)):
                line.append(columns[j][i])
            lines.append(line)
        return lines

    @torch.no_grad()
    def label2span(self, label_lst, prd_idx):
        column = []
        i = 0
        while (i < len(label_lst)):
            rela = label_lst[i]
            if ((i + 1) == prd_idx):
                column.append('(V*)')
                i += 1
            elif (rela == 'O'):
                column.append('*')
                i += 1
            else:
                position_tag = rela[0]
                label = rela[2:]
                if (position_tag in ('B', 'I')):
                    # 这里把I也考虑进来，防止第一个是I（I之前没有B，那么这个I当成B）
                    span_start = i
                    i += 1
                    labels = {}
                    labels[label] = 1
                    while (i < len(label_lst)):
                        if (label_lst[i][0] == 'I'):
                            labels[label_lst[i][2:]] = labels.get(
                                label_lst[i][2:], 0) + 1
                            i += 1
                        else:
                            # label_lst[i][0] == 'B' or 'O' 直接把i指向下一个或O
                            break
                    length = i - span_start
                    max_label = label
                    max_num = 0
                    for key, value in labels.items():
                        if (value > max_num):
                            max_num = value
                            max_label = key
                    if (length == 1):
                        column.append('(' + max_label + '*' + ')')
                    else:
                        column.append('(' + max_label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
        return column

    @torch.no_grad()
    def _predict(self, loader, file_name):
        self.model.eval()

        # f = open(file_name, 'w')
        preds = {}
        charts, probs = [], []
        bar = progress_bar(loader)
        sentence_idx = 0
        for idx, (words, *feats) in enumerate(bar):
            # pdb.set_trace()
            batch_size, _ = words.shape
            mask = words.ne(self.WORD.pad_index)
            t_mask = words.ne(self.WORD.pad_index)
            t_mask[:, 0] = 0
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
            # for i in range(batch_size):
            #     this_mask = t_mask[i]
            #     this_edge_score = s_edge[i]
            #     this_label_score = s_label[i]
            #     this_lemma = loader.dataset[sentence_idx].values[2]
            #     sentence_idx += 1
            #     lines = self.get_label_seqs(this_edge_score, this_label_score,
            #                                 this_mask, this_lemma)
            #     for line in lines:
            #         f.write(' '.join(line) + '\n')
            #     f.write('\n')

            edge_preds, label_preds = self.model.decode(s_edge, s_label)

            chart_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask),
                                                  -1)
            charts.extend(chart[1:i, :i].tolist()
                          for i, chart in zip(lens, chart_preds.unbind()))
            if self.args.prob:
                probs.extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        # f.close()
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls,
              path,
              optimizer_args={
                  'lr': 1e-3,
                  'betas': (.0, .95),
                  'eps': 1e-12,
                  'weight_decay': 3e-9
              },
              scheduler_args={'gamma': .75**(1 / 5000)},
              min_freq=7,
              fix_len=20,
              **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            optimizer_args (dict):
                Arguments for creating an optimizer.
            scheduler_args (dict):
                Arguments for creating a scheduler.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        interpolation = args.itp
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'lemma' in args.feat:
            LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
        if 'bert' in args.feat:
            from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            BERT.vocab = tokenizer.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)

        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        # SENSE = ChartField('senses', fn=CoNLL.get_srl_senses)

        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=(EDGE, LABEL))

        train = Dataset(transform, args.train)
        WORD.build(
            train, args.min_freq,
            (Embedding.load(args.embed, args.unk) if args.embed else None))
        if TAG is not None:
            TAG.build(train)
        if CHAR is not None:
            CHAR.build(train)
        if LEMMA is not None:
            LEMMA.build(train)
        LABEL.build(train)
        
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'interpolation': interpolation
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)


class VISemanticDependencyParser(BiaffineSemanticDependencyParser):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    NAME = 'vi-semantic-dependency'
    MODEL = VISemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), ChartMetric()

        for i, (words, *feats, edges, labels) in enumerate(bar, 1):
            # self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           s_label, edges, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            # self.optimizer.step()
            # self.scheduler.step()

            # edge_preds, label_preds = self.model.decode(s_edge, s_label)
            # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
            #        labels.masked_fill(~(edges.gt(0) & mask), -1))

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))

            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SrlMetric()

        for words, *feats, edges, labels in loader:
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            # loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
            #                                s_label, edges, labels, mask)
            loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           s_label, edges, labels, mask)
            # total_loss += loss.item()

            # edge_preds, label_preds = self.model.decode(s_edge, s_label)
            # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
            #        labels.masked_fill(~(edges.gt(0) & mask), -1))

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))

        # total_loss /= len(loader)

        # return total_loss, metric
        return metric

    @torch.no_grad()
    def _predict(self, loader, file_name):
        self.model.eval()

        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        for words, *feats in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
            s_edge = self.model.vi((s_edge, s_sib, s_cop, s_grd), mask)
            label_preds = self.model.decode(s_edge,
                                            s_label).masked_fill(~mask, -1)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            if self.args.prob:
                preds['probs'].extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens, s_edge.unbind())
                ])
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        return preds

    @classmethod
    def build(cls,
              path,
              optimizer_args={
                  'lr': 1e-3,
                  'betas': (.0, .95),
                  'eps': 1e-12
              },
              scheduler_args={'gamma': .75**(1 / 5000)},
              min_freq=7,
              fix_len=20,
              **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            optimizer_args (dict):
                Arguments for creating an optimizer.
            scheduler_args (dict):
                Arguments for creating a scheduler.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'lemma' in args.feat:
            LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
        if 'bert' in args.feat:
            from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            BERT.vocab = tokenizer.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=(EDGE, LABEL))

        train = Dataset(transform, args.train)
        WORD.build(
            train, args.min_freq,
            (Embedding.load(args.embed, args.unk) if args.embed else None))
        if TAG is not None:
            TAG.build(train)
        if CHAR is not None:
            CHAR.build(train)
        if LEMMA is not None:
            LEMMA.build(train)
        LABEL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'interpolation': args.itp
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)


class BiaffineSRLParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 20178.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    """

    NAME = 'biaffine-srl'
    MODEL = BiaffineSRLModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        # self.EDGE, self.LABEL = self.transform.PHEAD
        self.EDGE, self.LABEL, self.SENSE = self.transform.PHEAD

        self.transition_params = self.get_transition_params()

    def get_transition_params(self):
        label_strs = [
            self.LABEL.vocab[idx] for idx in range(len(self.LABEL.vocab))
        ] + ['O']
        num_tags = len(label_strs)
        transition_params = numpy.zeros([num_tags, num_tags],
                                        dtype=numpy.float32)
        for i, prev_label in enumerate(label_strs):
            for j, label in enumerate(label_strs):
                if i != j and label[
                        0] == 'I' and not prev_label == 'B' + label[1:]:
                    transition_params[i, j] = numpy.NINF
        return transition_params

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), ChartMetric()

        for words, *feats, edges, labels in bar:

            self.optimizer.zero_grad()
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            edge_preds, label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                   labels.masked_fill(~(edges.gt(0) & mask), -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SrlMetric()

        for words, *feats, edges, labels in loader:
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            # loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            # total_loss += loss.item()

            edge_preds, label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                   labels.masked_fill(~(edges.gt(0) & mask), -1))
        # total_loss /= len(loader)

        # return total_loss, metric
        return metric

    @torch.no_grad()
    def viterbi_decode(self, score):
        r'''
            This should only be used at test time.
            score: A [seq_len, num_tags] matrix of unary potentials.
                    seq_len is the real len
            viterbi: A [seq_len] list of integers containing the highest scoring tag
              indicies.
            viterbi_score: A float containing the score for the Viterbi sequence.
        '''
        score = numpy.array(score)
        transition_params = self.transition_params
        trellis = numpy.zeros_like(score)
        backpointers = numpy.zeros_like(score, dtype=numpy.int32)
        trellis[0] = score[0]
        for t in range(1, score.shape[0]):
            v = numpy.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + numpy.max(v, 0)
            backpointers[t] = numpy.argmax(v, 0)
        viterbi = [numpy.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = numpy.max(trellis[-1])
        return viterbi, viterbi_score

    @torch.no_grad()
    def get_label_seqs(self, s_edge, s_label, mask, lemmas):
        r'''
            get label_index seqs for a sentence
            s_egde (~torch.Tensor): ``[seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[ seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            mask :[seq_len] filter paded tokens and root
        '''
        seq_len, _, n_labels = s_label.shape
        real_len = mask.sum().item()
        # [seq_len, seq_len] tail,head
        edge_pred = s_edge.argmax(-1)
        # [seq_len] if token is a predicate
        predicate_mask = (edge_pred[:, 0] == 1) & mask
        predicate_num = predicate_mask.sum().item()
        # [seq_len, seq_len, n_labels] head, tail, n_labels
        s_label = s_label.permute(1, 0, 2)
        # [predicate_num, seq_len, n_labels]
        prd_seq_label_scores = s_label[predicate_mask]
        # [predicate_num, seq_len] whether exist edge between predicate and token
        prd_seq_mask = (edge_pred.permute(1, 0)[predicate_mask].eq(1)) & mask
        # [predicate_num, seq_len, n_labels+1]
        prd_seq_label_scores = torch.cat(
            (prd_seq_label_scores,
             torch.zeros_like(prd_seq_label_scores[..., :1])), -1)
        prd_seq_label_scores[..., -1] = float('-inf')
        prd_seq_label_scores.masked_fill_(
            (~prd_seq_mask).unsqueeze(2).expand(-1, -1, n_labels + 1),
            float('-inf'))
        end_mask = (~prd_seq_mask) & mask
        prd_seq_label_scores[end_mask, [-1] * end_mask.sum()] = 0.0
        # list [prd1_idx, prd2_idx] start from 1
        real_idx_prd = ((predicate_mask == True).nonzero()).squeeze(1).tolist()
        columns = []
        columns.append(predicate_mask.tolist()[1:real_len + 1])
        for w in prd_seq_label_scores:
            real_score = w.tolist()[1:real_len + 1]
            viterbi, viterbi_score = self.viterbi_decode(real_score)
            label_lst = []
            for idx in viterbi:
                if (idx < len(self.LABEL.vocab)):
                    label_lst.append(self.LABEL.vocab[idx])
                else:
                    label_lst.append('O')
            # pdb.set_trace()

            columns.append(label_lst)

        for i in range(1, len(columns)):
            prd_idx = real_idx_prd[i - 1]
            new_column = self.label2span(columns[i], prd_idx)
            columns[i] = new_column

        lines = []
        for i in range(real_len):
            line = []
            # print(i)
            # print(len(columns[0]))
            # print(mask)
            # print(mask.sum())
            if (columns[0][i]):
                line.append(lemmas[i])
            else:
                line.append('-')
            for j in range(1, len(columns)):
                line.append(columns[j][i])
            lines.append(line)
        return lines

    @torch.no_grad()
    def label2span(self, label_lst, prd_idx):
        column = []
        i = 0
        while (i < len(label_lst)):
            rela = label_lst[i]
            if ((i + 1) == prd_idx):
                column.append('(V*)')
                i += 1
            elif (rela == 'O'):
                column.append('*')
                i += 1
            else:
                position_tag = rela[0]
                label = rela[2:]
                if (position_tag in ('B', 'I')):
                    # 这里把I也考虑进来，防止第一个是I（I之前没有B，那么这个I当成B）
                    span_start = i
                    i += 1
                    labels = {}
                    labels[label] = 1
                    while (i < len(label_lst)):
                        if (label_lst[i][0] == 'I'):
                            labels[label_lst[i][2:]] = labels.get(
                                label_lst[i][2:], 0) + 1
                            i += 1
                        else:
                            # label_lst[i][0] == 'B' or 'O' 直接把i指向下一个或O
                            break
                    length = i - span_start
                    max_label = label
                    max_num = 0
                    for key, value in labels.items():
                        if (value > max_num):
                            max_num = value
                            max_label = key
                    if (length == 1):
                        column.append('(' + max_label + '*' + ')')
                    else:
                        column.append('(' + max_label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
        return column

    @torch.no_grad()
    def _predict(self, loader, file_name):
        self.model.eval()

        # f = open(file_name, 'w')
        preds = {}
        charts, probs = [], []
        bar = progress_bar(loader)
        sentence_idx = 0
        for idx, (words, *feats) in enumerate(bar):
            # pdb.set_trace()
            batch_size, _ = words.shape
            mask = words.ne(self.WORD.pad_index)
            t_mask = words.ne(self.WORD.pad_index)
            t_mask[:, 0] = 0
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
            # for i in range(batch_size):
            #     this_mask = t_mask[i]
            #     this_edge_score = s_edge[i]
            #     this_label_score = s_label[i]
            #     this_lemma = loader.dataset[sentence_idx].values[2]
            #     sentence_idx += 1
            #     lines = self.get_label_seqs(this_edge_score, this_label_score,
            #                                 this_mask, this_lemma)
            #     for line in lines:
            #         f.write(' '.join(line) + '\n')
            #     f.write('\n')

            edge_preds, label_preds = self.model.decode(s_edge, s_label)

            chart_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask),
                                                  -1)
            charts.extend(chart[1:i, :i].tolist()
                          for i, chart in zip(lens, chart_preds.unbind()))
            if self.args.prob:
                probs.extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        # f.close()
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls,
              path,
              optimizer_args={
                  'lr': 1e-3,
                  'betas': (.0, .95),
                  'eps': 1e-12,
                  'weight_decay': 3e-9
              },
              scheduler_args={'gamma': .75**(1 / 5000)},
              min_freq=7,
              fix_len=20,
              **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            optimizer_args (dict):
                Arguments for creating an optimizer.
            scheduler_args (dict):
                Arguments for creating a scheduler.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        interpolation = args.itp
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'lemma' in args.feat:
            LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
        if 'bert' in args.feat:
            from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            BERT.vocab = tokenizer.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)

        LABEL = ChartField('labels', fn=CoNLL.get_srl_labels)
        SENSE = ChartField('senses', fn=CoNLL.get_srl_senses)

        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=(EDGE, LABEL, SENSE))

        train = Dataset(transform, args.train)
        WORD.build(
            train, args.min_freq,
            (Embedding.load(args.embed, args.unk) if args.embed else None))
        if TAG is not None:
            TAG.build(train)
        if CHAR is not None:
            CHAR.build(train)
        if LEMMA is not None:
            LEMMA.build(train)
        LABEL.build(train)

        SENSE.build(train)
        
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_senses': len(SENSE.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'interpolation': interpolation
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)
