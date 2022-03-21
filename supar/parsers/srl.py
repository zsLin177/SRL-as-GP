# -*- coding: utf-8 -*-

import os

import pdb

import torch
import torch.nn as nn
from supar.models import (BiaffineSemanticRoleLabelingModel,
                          VISemanticRoleLabelingModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric, SrlMetric
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class BiaffineSemanticRoleLabelingParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser (:cite:`dozat-etal-2018-simpler`).
    """

    NAME = 'biaffine-semantic-role-labeling'
    MODEL = BiaffineSemanticRoleLabelingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
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
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
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

        for i, (words, *feats, labels) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_label = self.model(words, feats, if_prd=if_prd)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SrlMetric()

        for words, *feats, labels in loader:
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_label = self.model(words, feats, if_prd=if_prd)
            # loss = self.model.loss(s_edge, s_label, labels, mask)
            # total_loss += loss.item()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'labels': [], 'probs': [] if self.args.prob else None}

        strans, trans, B_idxs, I_idxs, prd_idx, pair_dict = self.prepare_viterbi()
        if(torch.cuda.is_available()):
            strans = strans.cuda()
            trans = trans.cuda()
        for words, *feats, labels in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            n_mask = mask[:, :, 0]
            lens = mask[:, 1].sum(-1).tolist()
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_label = self.model(words, feats, if_prd=if_prd)
            if(not self.args.vtb):
                label_preds = self.model.decode(s_edge,
                                                s_label).masked_fill(~mask, -1)
                if self.args.given_prd:
                    label_preds[:, :, 0] = labels[:, :, 0]
                    prd_mask = labels[:, :, 0].eq(prd_idx)
                    seq_len = label_preds.shape[1]
                    prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                    prd_mask_2[:, :, 0] = 1
                    label_preds.masked_fill_(~prd_mask_2, -1)
            else:
                edge_preds, label_preds, p_label = self.model.viterbi_decode3(s_edge, s_label, strans, trans, n_mask, mask, B_idxs, I_idxs, prd_idx)
                label_preds.masked_fill_(~mask, -1)
                if self.args.given_prd:
                    label_preds[:, :, 0] = labels[:, :, 0]
                    prd_mask = labels[:, :, 0].eq(prd_idx)
                    seq_len = label_preds.shape[1]
                    prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                    prd_mask_2[:, :, 0] = 1
                    label_preds.masked_fill_(~prd_mask_2, -1)
            label_preds = self.model.fix_label_cft(label_preds, B_idxs, I_idxs, prd_idx, pair_dict, p_label)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            if self.args.prob:
                preds['probs'].extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        return preds

    def prepare_viterbi(self):
        # [n_labels+2]
        strans = [0] * (len(self.LABEL.vocab)+2)
        trans = [[0] * (len(self.LABEL.vocab)+2) for _ in range((len(self.LABEL.vocab)+2))]
        B_idxs = []
        I_idxs = []
        pair_dict = {}
        B2I_dict = {}
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                strans[i] = -float('inf')  # cannot start with I-
                I_idxs.append(i)
            elif(label.startswith('B-')):
                B_idxs.append(i)
                B2I_dict[label[2:]] = [i]
            elif(label == '[prd]'):
                # label = [prd]
                strans[i] = -float('inf')
                trans[i] = [-float('inf')] * (len(self.LABEL.vocab)+2)
                for j in range(len(trans)):
                    trans[j][i] = -float('inf')
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
    
        for idx, label in enumerate(self.LABEL.vocab.itos):
            if label == '[prd]':
                continue
            position_label = label[0:2]
            real_label = label[2:]
            if position_label == 'B-':
                pair_p_label = 'I-'
            else:
                pair_p_label = 'B-'
            pair_label = pair_p_label+real_label
            if pair_label in self.LABEL.vocab.itos:
                pair_dict[idx] = self.LABEL.vocab.itos.index(pair_label)
            else:
                # for some label, they only have B-x
                pair_dict[idx] = idx

        # for key, value in B2I_dict.items():
        #     if(len(value)>1):
        #         b_idx = value[0]
        #         i_idx = value[1]
        #         for idx in I_idxs:
        #             trans[b_idx][idx] = -float('inf')
        #         trans[b_idx][i_idx] = 0

        # for i in B_idxs:
        #     trans[i][-1] = -float('inf')
        for i in B_idxs:
            trans[-2][i] = -float('inf')

        for i in I_idxs:
            for j in I_idxs:
                trans[i][j] = -float('inf')
            trans[i][-2] = -float('inf')
        trans[-2][-1] = -float('inf')
        trans[-1][-2] = -float('inf')
        for i in I_idxs:
            trans[-1][i] = -float('inf')

        strans[-2] = -float('inf')
        
        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, self.LABEL.vocab.stoi['[prd]'], pair_dict

    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
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
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                fix_len=args.fix_len,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
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
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
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
            'n_words':
            len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'unk_index':
            WORD.unk_index,
            'bos_index':
            WORD.bos_index,
            'lr':
            5e-5,
            'epochs': 10, 
            'warmup':
            0.1,
            'interpolation': args.itp,
            'split': args.split,
            'gold_p': args.given_prd
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class VISemanticRoleLabelingParser(BiaffineSemanticRoleLabelingParser):
    r"""
    The implementation of SRL Parser using Variational Inference (:cite:`wang-etal-2019-second`).
    """

    NAME = 'vi-semantic-role-labeling'
    MODEL = VISemanticRoleLabelingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD
        self.strans, self.trans, self.B_idxs, self.I_idxs, self.prd_idx, self.pair_dict = self.prepare_viterbi()

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
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
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
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

        for i, (words, *feats, labels) in enumerate(bar, 1):
            word_mask = words.ne(self.args.pad_index)
            # mask2 = word_mask.clone()
            # mask2[:, 0] = 0
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats, if_prd)
            # pdb.set_trace()
            loss, s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            label_preds = self.model.decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1),
                   labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader, given_prd=False):
        self.model.eval()

        total_loss, metric = 0, SrlMetric()

        for words, *feats, labels in loader:
            word_mask = words.ne(self.args.pad_index)
            # mask2 = word_mask.clone()
            # mask2[:, 0] = 0
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats, if_prd)
            s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, labels, mask, if_pred=True)
            # total_loss += loss.item()

            if(not given_prd):
                label_preds = self.model.decode(s_edge, s_label)
                metric(label_preds.masked_fill(~mask, -1),
                    labels.masked_fill(~mask, -1))
            else:
                prd_mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                prd_mask[:, 0] = 0
                prd_mask = prd_mask & labels[:, :, 0].gt(-1)
                label_preds = self.model.decode(s_edge, s_label)
                label_preds[:, :, 0] = labels[:, :, 0]
                seq_len = labels.shape[1]
                # pdb.set_trace()
                prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                prd_mask_2[:, :, 0] = 1
                label_preds.masked_fill_(~prd_mask_2, -1)
                metric(label_preds.masked_fill(~mask, -1),
                    labels.masked_fill(~mask, -1))

        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        strans, trans, B_idxs, I_idxs, prd_idx, pair_dict = self.prepare_viterbi()
        if(torch.cuda.is_available()):
            strans = strans.cuda()
            trans = trans.cuda()
        for words, *feats, labels in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            # mask2 = word_mask.clone()
            # mask2[:, 0] = 0
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            n_mask = mask[:, :, 0]
            lens = mask[:, 1].sum(-1).tolist()
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats, if_prd)
            # s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
            s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, labels, mask, True)
    
            label_preds, p_label = self.model.viterbi_decode3(s_edge, s_label, strans, trans, n_mask, mask, B_idxs, I_idxs, prd_idx)
            label_preds.masked_fill_(~mask, -1)

            if self.args.given_prd:
                    label_preds[:, :, 0] = labels[:, :, 0]
                    prd_mask = labels[:, :, 0].eq(prd_idx)
                    seq_len = label_preds.shape[1]
                    prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                    prd_mask_2[:, :, 0] = 1
                    label_preds.masked_fill_(~prd_mask_2, -1)
            
            label_preds = self.model.fix_label_cft(label_preds, B_idxs, I_idxs, prd_idx, pair_dict, p_label)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        return preds

    
    @torch.no_grad()
    def _api(self, loader, prd_idxs=None, task='09'):
        self.model.eval()
        if(task != '09'):
            strans, trans, B_idxs, I_idxs, prd_idx = self.strans, self.trans, self.B_idxs, self.I_idxs, self.prd_idx
            batch_tuple_lists = []
            if(torch.cuda.is_available()):
                strans = strans.cuda()
                trans = trans.cuda()
            for words, *feats, labels in progress_bar(loader):
                word_mask = words.ne(self.args.pad_index)
                # mask2 = word_mask.clone()
                # mask2[:, 0] = 0
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                n_mask = mask[:, :, 0]
                lens = mask[:, 1].sum(-1).tolist()
                s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats)
                # s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
                s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                            x, labels, mask, True)
                # if(not self.args.vtb):
                #     label_preds = self.model.decode(s_edge,
                #                                 s_label).masked_fill(~mask, -1)
                # else:
                label_preds = self.model.viterbi_decode3(s_edge, s_label, strans, trans, n_mask, mask, B_idxs, I_idxs, prd_idx)
                tuples = self.build_spans(label_preds)
                batch_tuple_lists.append(tuples)
        else:
            batch_tuple_lists = []
            for words, *feats, labels in progress_bar(loader):
                word_mask = words.ne(self.args.pad_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats)
                s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                            x, labels, mask, True)
                if prd_idxs == None :
                    label_preds = self.model.decode(s_edge, s_label)
                else:
                    prd_mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                    prd_mask[:, 0] = 0
                    for i, lst in enumerate(prd_idxs):
                        if(len(lst) == 0):
                            continue
                        else:
                            for j in lst:
                                prd_mask[i, j] = 1
                    label_preds = self.model.decode(s_edge, s_label)
                    prd_labels = -torch.ones_like(prd_mask, dtype=torch.long)
                    prd_labels.masked_fill_(prd_mask, 0)
                    label_preds[:, :, 0] = prd_labels
                    seq_len = label_preds.shape[1]
                    prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                    prd_mask_2[:, :, 0] = 1
                    label_preds.masked_fill_(~prd_mask_2, -1)

                tuples = self.build_spans(label_preds, task)
                batch_tuple_lists.append(tuples)

        return batch_tuple_lists

    
    def build_spans(self, label_pred, task='05'):
        # label_pred: [batch_size, seq_len, seq_len]
        # return [batch_size, seq_len, seq_len, seq_len] -> [tuples] tuple:[sent_idx_inbatch, pred_idx, span_start_idx, span_end_idx, label]
        if(task != '09'): 
            prd_idx = self.prd_idx

            batch_size, seq_len = label_pred.shape[0], label_pred.shape[1]
            # [batch_size, seq_len]
            pred_mask = label_pred[..., 0].ne(-1)
            pred_mask[:, 0] = 0

            k = pred_mask.sum()
            if(k <= 0):
                return []
            
            pred_idxs = pred_mask.nonzero()
            batch_idx = pred_idxs[:, 0]
            pred_word_idx = pred_idxs[:, 1]
            # [k, seq_len]
            predicate_label_seq = label_pred[batch_idx, :, pred_word_idx]
            predicate_label_seq = predicate_label_seq.masked_fill(predicate_label_seq.eq(prd_idx), -1)

            lst = predicate_label_seq.tolist()
            b_idx = []
            s_idx = []
            e_idx = []
            span_label_idx = []
            for i in range(k):
                seq = lst[i][1:]  # [seq_len-1]
                length = len(seq)
                j = 0
                while(j < length):
                    if(seq[j] == -1):
                        j += 1
                    elif(seq[j] in self.I_idxs):
                        j += 1  # delete conflict I (it is so helpful)
                        # maybe set a gap p(I_idx)>0.5 
                    else:
                        span_start = j
                        span_end = -1
                        label1_idx = seq[j]
                        label1 = self.LABEL.vocab.itos[label1_idx][2:]
                        j += 1
                        while (j < length):
                            if(seq[j] == -1):
                                j += 1
                            elif(seq[j] in self.B_idxs):
                                break
                            else:
                                span_end = j
                                label2_idx = seq[j]
                                label2 = self.LABEL.vocab.itos[label2_idx][2:]
                                j += 1
                                break
                        
                        if(span_end != -1):
                            if(label1 == label2):
                                # 前后不一样的删去
                                s_idx.append(span_start+1)
                                e_idx.append(span_end+1)
                                span_label_idx.append(label1_idx)
                                b_idx.append(i)
                        else:
                            s_idx.append(span_start+1)
                            e_idx.append(span_start+1)
                            b_idx.append(i)
                            span_label_idx.append(label1_idx)

            k_spans = -torch.ones((k, seq_len, seq_len), device=label_pred.device).long()
            k_spans_mask = k_spans.gt(-1)
            k_spans_mask[b_idx, s_idx, e_idx] = True
            k_spans = k_spans.masked_scatter(k_spans_mask, k_spans.new_tensor(span_label_idx))

            back_mask = pred_mask.unsqueeze(-1).expand(-1, -1, seq_len).unsqueeze(-1).expand(-1, -1, -1,seq_len)
            spans = -torch.ones((batch_size, seq_len, seq_len, seq_len), device=label_pred.device).long()
            spans = spans.masked_scatter(back_mask, k_spans)
            label_mask = spans.gt(-1)
            res_lists = label_mask.nonzero(as_tuple=False).tolist()
            label_idxs = spans[label_mask].tolist()
            labels_list = [self.LABEL.vocab.itos[t_idx][2:] for t_idx in label_idxs]
            for i in range(len(res_lists)):
                res_lists[i].append(labels_list[i])

            tmp_dict = {}
            final_lst = []
            for tup in res_lists:
                if(tup[1] in tmp_dict):
                    final_lst[tmp_dict[tup[1]]].append(tup)
                else:
                    tmp_dict[tup[1]] = len(tmp_dict)
                    final_lst.append([tup])
            
            for predicate_lst in final_lst:
                del_rept_dict = {"ARG0":0, "ARG1":0, "ARG2":0, "ARG3":0, "ARG4":0}
                for i in range(len(predicate_lst)-1, -1, -1):
                    tup = predicate_lst[i]
                    label = tup[-1]
                    if(label in del_rept_dict and del_rept_dict[label] == 0):
                        del_rept_dict[label] = 1
                    elif(label in del_rept_dict and del_rept_dict[label] == 1):
                        predicate_lst.pop(i)
            return final_lst
        else:
            label_pred[:, 0] = -1
            label_pred[:, :, 0] = -1
            label_pred = label_pred.transpose(1, 2)
            all_idxs = label_pred.gt(-1).nonzero().tolist()
            for i in range(len(all_idxs)):
                lst = all_idxs[i]
                b_id, p_id, c_id = lst
                label = self.LABEL.vocab.itos[label_pred[b_id, p_id, c_id].item()]
                all_idxs[i].append(c_id)
                all_idxs[i].append(label)
            return all_idxs


    def prepare_viterbi(self):
        # [n_labels+2]
        strans = [0] * (len(self.LABEL.vocab)+2)
        trans = [[0] * (len(self.LABEL.vocab)+2) for _ in range((len(self.LABEL.vocab)+2))]
        B_idxs = []
        I_idxs = []
        pair_dict = {}
        B2I_dict = {}
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                strans[i] = -float('inf')  # cannot start with I-
                I_idxs.append(i)
            elif(label.startswith('B-')):
                B_idxs.append(i)
                B2I_dict[label[2:]] = [i]
            elif(label == '[prd]'):
                # label = [prd]
                strans[i] = -float('inf')
                trans[i] = [-float('inf')] * (len(self.LABEL.vocab)+2)
                for j in range(len(trans)):
                    trans[j][i] = -float('inf')
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
    
        for idx, label in enumerate(self.LABEL.vocab.itos):
            if label == '[prd]':
                continue
            position_label = label[0:2]
            real_label = label[2:]
            if position_label == 'B-':
                pair_p_label = 'I-'
            else:
                pair_p_label = 'B-'
            pair_label = pair_p_label+real_label
            if pair_label in self.LABEL.vocab.itos:
                pair_dict[idx] = self.LABEL.vocab.itos.index(pair_label)
            else:
                # for some label, they only have B-x
                pair_dict[idx] = idx

        # for key, value in B2I_dict.items():
        #     if(len(value)>1):
        #         b_idx = value[0]
        #         i_idx = value[1]
        #         for idx in I_idxs:
        #             trans[b_idx][idx] = -float('inf')
        #         trans[b_idx][i_idx] = 0

        # for i in B_idxs:
        #     trans[i][-1] = -float('inf')
        for i in B_idxs:
            trans[-2][i] = -float('inf')

        for i in I_idxs:
            for j in I_idxs:
                trans[i][j] = -float('inf')
            trans[i][-2] = -float('inf')
        trans[-2][-1] = -float('inf')
        trans[-1][-2] = -float('inf')
        for i in I_idxs:
            trans[-1][i] = -float('inf')

        strans[-2] = -float('inf')
        
        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, self.LABEL.vocab.stoi['[prd]'], pair_dict


    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
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
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                fix_len=args.fix_len,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
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
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
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
            'n_words':
            len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'unk_index':
            WORD.unk_index,
            'bos_index':
            WORD.bos_index,
            'lr':
            5e-5,
            'epochs': 10, 
            'warmup':
            0.1,
            'interpolation': args.itp,
            'split': args.split,
            'gold_p': args.given_prd
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)
