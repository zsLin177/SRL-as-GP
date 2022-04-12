from enum import EnumMeta
import os
import pdb
import subprocess
import torch
import torch.nn as nn
from torch.nn import parameter
from torch.serialization import load
from supar import models
from supar.models import (BiaffineSrlModel,
                          VISrlModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField, SpanSrlFiled, ElmoField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.logging import init_logger, logger
from supar.utils.metric import ChartMetric, SrlMetric
from supar.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler

logger = get_logger(__name__)

class VLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=8000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(VLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        scale = min(pow(epoch, -0.5), epoch * pow(self.warmup_steps, -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class BiaffineSrlParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 20178.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    """

    NAME = 'biaffine-semantic-role-labeling'
    MODEL = BiaffineSrlModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.WORD, self.CHAR, self.BERT = self.transform.FORM

        self.WORD, self.CHAR, self.ELMO, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        # self.EDGE, self.LABEL, self.SPAN = self.transform.PHEAD
        self.EDGE, self.LABEL = self.transform.PHEAD
        # self.bi2label, self.I_idxs, self.B_idxs = self.idx_prepare()
        
    def idx_prepare(self):
        bi2label = {}
        I_idxs = []
        B_idxs = []
        for i, s in enumerate(self.LABEL.vocab.itos):
            if(s.startswith('I-')):
                I_idxs.append(i)
                bi2label[i] = self.SPAN.vocab.stoi[s[2:]]
            elif(s.startswith('B-')):
                B_idxs.append(i)
                bi2label[i] = self.SPAN.vocab.stoi[s[2:]]
        return bi2label, I_idxs, B_idxs

    def train(self,
              train,
              dev,
              test,
              dev_pred=None,
              dev_gold=None,
              test_pred=None,
              test_gold=None,
              buckets=32,
              batch_size=5000,
              clip=5.0,
              epochs=5000,
              patience=100,
              **kwargs):
        
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
            # if(self.args.repr_gold):
            s_edge, s_label = self.model(words, feats, edges)
            # else:
            #     s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

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
        # metric2 = ChartMetric()

        for words, *feats, edges, labels in loader:
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(words, feats)
            # loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            # total_loss += loss.item()

            edge_preds, label_preds = self.model.decode(s_edge, s_label)
            label_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1)
            metric(label_preds,
                   labels.masked_fill(~(edges.gt(0) & mask), -1))
            # metric2(self.build_spans(label_preds), spans)
        # total_loss /= len(loader)

        # return total_loss, metric
        # return metric, metric2
        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        charts, probs = [], []
        spans_lst = []
        print("Total number of paramerters in model is {} M".format(sum(x.numel() for x in self.model.parameters())/1e6))
        strans, trans, B_idxs, I_idxs, prd_idx, pair_dict = self.prepare_viterbi()
        # strans, trans = self.prepare_viterbi2()
        if(torch.cuda.is_available()):
            strans = strans.cuda()
            trans = trans.cuda()
        for words, *feats, edges, labels in progress_bar(loader):
            batch_size = words.shape[0]
            mask = words.ne(self.WORD.pad_index)
            n_mask = mask.clone()
            n_mask[:, 0] = 0
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
            if(not self.args.vtb):
                edge_preds, label_preds = self.model.decode(s_edge, s_label)
                label_preds.masked_fill_(~mask, -1)
                if self.args.given_prd:
                    label_preds[:, :, 0] = labels[:, :, 0]
                    prd_mask = labels[:, :, 0].eq(prd_idx)
                    seq_len = label_preds.shape[1]
                    prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                    prd_mask_2[:, :, 0] = 1
                    label_preds.masked_fill_(~prd_mask_2, -1)
                # chart_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask),
                #                                   -1)
                chart_preds = label_preds
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
                chart_preds = label_preds

            charts.extend(chart[1:i, :i].tolist()
                          for i, chart in zip(lens, chart_preds.unbind()))
            if self.args.prob:
                probs.extend([
                    prob[1:i, :i].cpu()
                    for i, prob in zip(lens,
                                       s_edge.softmax(-1).unbind())
                ])
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs

        # return preds, spans_lst
        return preds


    def build_spans(self, label_pred):
        # label_pred: [batch_size, seq_len, seq_len]
        # return [batch_size, seq_len, seq_len, seq_len]
        prd_idx = self.LABEL.vocab.stoi['[prd]']

        batch_size, seq_len = label_pred.shape[0], label_pred.shape[1]
        # [batch_size, seq_len]
        pred_mask = label_pred[..., 0].ne(-1)
        pred_mask[:, 0] = 0

        k = pred_mask.sum()
        if(k <= 0):
            return -torch.ones((batch_size, seq_len, seq_len, seq_len), device=label_pred.device).long()
        
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
                    label1 = self.bi2label[seq[j]]
                    j += 1
                    while (j < length):
                        if(seq[j] == -1):
                            j += 1
                        elif(seq[j] in self.B_idxs):
                            break
                        else:
                            span_end = j
                            label2 = self.bi2label[seq[j]]
                            j += 1
                            break
                    
                    if(span_end != -1):
                        if(label1 == label2):
                            # 前后不一样的删去
                            s_idx.append(span_start+1)
                            e_idx.append(span_end+1)
                            span_label_idx.append(label1)
                            b_idx.append(i)
                    else:
                        s_idx.append(span_start+1)
                        e_idx.append(span_start+1)
                        b_idx.append(i)
                        span_label_idx.append(label1)

        k_spans = -torch.ones((k, seq_len, seq_len), device=label_pred.device).long()
        k_spans_mask = k_spans.gt(-1)
        k_spans_mask[b_idx, s_idx, e_idx] = True
        k_spans = k_spans.masked_scatter(k_spans_mask, k_spans.new_tensor(span_label_idx))

        back_mask = pred_mask.unsqueeze(-1).expand(-1, -1, seq_len).unsqueeze(-1).expand(-1, -1, -1,seq_len)
        spans = -torch.ones((batch_size, seq_len, seq_len, seq_len), device=label_pred.device).long()
        spans = spans.masked_scatter(back_mask, k_spans)

        return spans

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

    def prepare_viterbi2(self):
        # [n_labels+2]
        strans = [0] * len(self.LABEL.vocab)
        trans = [[0] * (len(self.LABEL.vocab)) for _ in range(len(self.LABEL.vocab))]
        B_idxs = []
        I_idxs = []
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
                trans[i] = [-float('inf')] * len(self.LABEL.vocab)
                for j in range(len(trans)):
                    trans[j][i] = -float('inf')
        
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
    
        # for key, value in B2I_dict.items():
        #     if(len(value)>1):
        #         b_idx = value[0]
        #         i_idx = value[1]
        #         for idx in I_idxs:
        #             trans[b_idx][idx] = -float('inf')
        #         trans[b_idx][i_idx] = 0

        for i in I_idxs:
            for j in I_idxs:
                trans[i][j] = -float('inf')
        
        # pdb.set_trace()
        return torch.tensor(strans), torch.tensor(trans)

    def prepare_viterbi3(self):
        # for biiio
        strans = [0] * (len(self.LABEL.vocab)+1)
        trans = [[0] * (len(self.LABEL.vocab)+1) for _ in range((len(self.LABEL.vocab)+1))]
        B_idxs = []
        I_idxs = []
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
                trans[i] = [-float('inf')] * (len(self.LABEL.vocab)+1)
                for j in range(len(trans)):
                    trans[j][i] = -float('inf')
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
        
        for i in I_idxs:
            trans[-1][i] = -float('inf')
        
        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, self.LABEL.vocab.stoi['[prd]']
    
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
        TAG, CHAR, LEMMA, BERT, ELMO = None, None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'elmo' in args.feat:
            ELMO = ElmoField('elmo')
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
        # SPAN = SpanSrlFiled('spans', build_fn=CoNLL.get_span_labels, fn=CoNLL.get_spans)
        # transform = CoNLL(FORM=(WORD, CHAR, BERT),
        #                   LEMMA=LEMMA,
        #                   POS=TAG,
        #                   PHEAD=(EDGE, LABEL, SPAN))
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT),
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
        if(args.use_pred):
            LABEL.vocab.extend(['Other'])
        # SPAN.build(train)
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
            'interpolation': interpolation,
            'encoder': args.encoder,
            'elmo_dropout': args.elmo_dropout
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")
        if (args.encoder != 'transformer'):
            optimizer = Adam(model.parameters(), **optimizer_args)
            scheduler = ExponentialLR(optimizer, **scheduler_args)
        else:
            optimizer = Adam(model.parameters(),
                             lr=0.04,
                             betas=(0.9, 0.98),
                             eps=1e-12)
            scheduler = VLR(optimizer, warmup_steps=8000)

        return cls(args, model, transform, optimizer, scheduler)

class VISrlParser(BiaffineSrlParser):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    NAME = 'vi-semantic-role-labeling'
    MODEL = VISrlModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.ELMO, self.BERT = self.transform.FORM
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
            mask = words.ne(self.WORD.pad_index)
            mask2 = mask.clone()
            mask2[:, 0] = 0
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats)
            loss, s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, edges, labels, mask, mask2)
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
            mask2 = mask.clone()
            mask2[:, 0] = 0
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats)
            # loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd,
            #                                s_label, edges, labels, mask)
            loss, s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, edges, labels, mask, mask2, True)
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
    def _predict(self, loader):

        self.model.eval()
        print("Total number of paramerters in model is {} M".format(sum(x.numel() for x in self.model.parameters())/1e6))
        preds = {'labels': [], 'probs': [] if self.args.prob else None}

        pred_sum = 0
        con_sum = 0
        if self.args.schema in ('BE', 'BII', 'BIES', 'BES'):
            if self.args.schema == 'BE':
                strans, trans, B_idxs, I_idxs, prd_idx, pair_dict = self.prepare_viterbi_BE()
            elif self.args.schema == 'BII':
                strans, trans, B_idxs, I_idxs, prd_idx = self.prepare_viterbi_BII()
            elif self.args.schema == 'BIES':
                strans, trans, B_idxs, I_idxs, E_idxs, S_idxs, prd_idx = self.prepare_viterbi_BIES()
            elif self.args.schema == 'BES':
                strans, trans, B_idxs, E_idxs, S_idxs, pair_dict, prd_idx = self.prepare_viterbi_BES()
            if(torch.cuda.is_available()):
                strans = strans.cuda()
                trans = trans.cuda()
        for words, *feats, edges, labels in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask2 = word_mask.clone()
            mask2[:, 0] = 0
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_sib, s_cop, s_grd, x = self.model(words, feats)
            # s_edge = self.model.vi((s_edge, s_sib, s_cop, s_grd), mask)
            loss, s_edge, s_label = self.model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, edges, labels, mask, mask2, True)
            if self.args.schema in ('BE', 'BII', 'BIES', 'BES'):
                if(not self.args.vtb):
                    label_preds = self.model.decode(s_edge,
                                                s_label).masked_fill(~mask, -1)
                    if self.args.given_prd:
                        # use gold predicate to constrain
                        label_preds[:, :, 0] = labels[:, :, 0]
                        prd_mask = labels[:, :, 0].eq(prd_idx)
                        seq_len = label_preds.shape[1]
                        prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                        prd_mask_2[:, :, 0] = 1
                        label_preds.masked_fill_(~prd_mask_2, -1)
                else:
                    if self.args.schema == 'BE':
                        label_preds, p_num, con_p_num, p_label = self.model.viterbi_decode_BE(s_edge, s_label, strans, trans, mask2, mask, B_idxs, I_idxs, pair_dict, prd_idx)
                    elif self.args.schema == 'BII':
                        label_preds, p_num, con_p_num, p_label = self.model.viterbi_decode_BII(s_edge, s_label, strans, trans, mask2, mask, B_idxs, I_idxs, prd_idx)
                    elif self.args.schema == 'BIES':
                        label_preds, p_num, con_p_num, p_label = self.model.viterbi_decode_BIES(s_edge, s_label, strans, trans, mask2, mask, B_idxs, I_idxs, E_idxs, S_idxs, prd_idx)
                    else:
                        label_preds, p_num, con_p_num, p_label = self.model.viterbi_decode_BES(s_edge, s_label, strans, trans, mask2, mask, B_idxs, E_idxs, S_idxs, pair_dict, prd_idx)
                    label_preds.masked_fill_(~mask, -1)
                    if self.args.given_prd:
                        # use gold predicate to constrain
                        label_preds[:, :, 0] = labels[:, :, 0]
                        prd_mask = labels[:, :, 0].eq(prd_idx)
                        seq_len = label_preds.shape[1]
                        prd_mask_2 = prd_mask.unsqueeze(-1).expand(-1, -1, seq_len).transpose(1, 2).clone()
                        prd_mask_2[:, :, 0] = 1
                        label_preds.masked_fill_(~prd_mask_2, -1)
                    if self.args.schema == 'BE':
                        label_preds = self.model.fix_label_cft_BE(label_preds, B_idxs, I_idxs, prd_idx, pair_dict, p_label)
                    elif self.args.schema == 'BES':
                        label_preds = self.model.fix_label_cft_BES(label_preds, B_idxs, E_idxs, S_idxs, pair_dict,prd_idx, p_label)
            else:
                # schema == SIM
                # this schema has the constrain that cannot be fixed currently
                # TODO
                label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
                if self.args.given_prd:
                    # use gold predicate to constrain
                    label_preds[:, :, 0] = labels[:, :, 0]

            preds['labels'].extend(chart[1:i+1, :i+1].tolist()
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
        # print('sum of predicted predicates:', pred_sum, 'sum of conflicting predicates:', con_sum, end=' ')
        # print('ratio of conficting predicates:', con_sum/pred_sum)

        return preds

    def prepare_viterbi_BE(self):
        '''
        for BE schema
        '''
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

        for key, value in B2I_dict.items():
            if(len(value)>1):
                b_idx = value[0]
                i_idx = value[1]
                for idx in I_idxs:
                    trans[b_idx][idx] = -float('inf')
                trans[b_idx][i_idx] = 0

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

    @torch.no_grad()
    def prepare_detail_vtb(self, loader):
        trans_sum = [0] * (len(self.LABEL.vocab)-1)
        trans = [[0] * (len(self.LABEL.vocab)-1) for _ in range((len(self.LABEL.vocab)-1))]
        
        for words, *feats, edges, labels in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            labels = labels.masked_fill(~mask, -1)
            # [batch_size, seq_len]
            pred_mask = labels[:, :, 0].eq(self.LABEL.vocab.stoi['[prd]'])
            # [k, seq_len]
            edge_label_seq = labels.transpose(1, 2)[pred_mask]
            k = edge_label_seq.shape[0]
            edge_label_seq = edge_label_seq.tolist()
            for i in range(k):
                tmp = [t for t in edge_label_seq[i] if t > -1]
                if(len(tmp)<=1):
                    continue
                for j in range(0, len(tmp)-1):
                    trans_sum[tmp[j]] += 1
                    trans[tmp[j]][tmp[j+1]] += 1
        
        trans_sum, trans = torch.tensor(trans_sum), torch.tensor(trans)
        trans_sum = trans_sum.masked_fill(trans_sum.eq(0), 1)
        log_trans_pro = (trans/(trans_sum.unsqueeze(1))).log()
        return log_trans_pro

    def prepare_viterbi2_1(self):
        # directly process edge sequence
        strans = [0] * (len(self.LABEL.vocab)-1)
        trans = [[0] * (len(self.LABEL.vocab)-1) for _ in range((len(self.LABEL.vocab)-1))]
        B_idxs = []
        I_idxs = []
        B2I_dict = {}
        pair_dict = {}
        single_idxs = []
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                strans[i] = -float('inf')  # cannot start with I-
                I_idxs.append(i)
            elif(label.startswith('B-')):
                B_idxs.append(i)
                B2I_dict[label[2:]] = [i]
            elif(label == '[prd]'):
                # label = [prd]
                continue
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
    
        for real_label in B2I_dict:
            if(len(B2I_dict[real_label]) == 2):
                idx1, idx2 = B2I_dict[real_label][0], B2I_dict[real_label][1]
                pair_dict[idx1] = idx2
                pair_dict[idx2] = idx1
            else:
                single_idxs.extend(B2I_dict[real_label])

        # forbid I->I
        for i in I_idxs:
            for j in I_idxs:
                trans[i][j] = -float('inf')

        # forbid b-a1->i-a0
        for b_idx in B_idxs:
            if(b_idx in single_idxs):
                # 只能是由一个词组成的论元,
                for i_idx in I_idxs:
                    trans[b_idx][i_idx] = -float('inf')
            else:
                # 可以由多个词组成的论元
                for i_idx in I_idxs:
                    trans[b_idx][i_idx] = -float('inf')
                trans[b_idx][pair_dict[b_idx]] = 0
        
        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, self.LABEL.vocab.stoi['[prd]'], pair_dict, single_idxs

    def prepare_viterbi_BII(self):
        '''
        for BII schema
        '''
        # for biiio
        strans = [0] * (len(self.LABEL.vocab)+1)
        trans = [[0] * (len(self.LABEL.vocab)+1) for _ in range((len(self.LABEL.vocab)+1))]
        B_idxs = []
        I_idxs = []
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
                trans[i] = [-float('inf')] * (len(self.LABEL.vocab)+1)
                for j in range(len(trans)):
                    trans[j][i] = -float('inf')
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                real_label = label[2:]
                if(real_label in B2I_dict):
                    B2I_dict[real_label].append(i)
        
        for i in I_idxs:
            # cannot trans from 'O' to 'I'
            trans[-1][i] = -float('inf')
        
        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, self.LABEL.vocab.stoi['[prd]']
    
    def prepare_viterbi_BIES(self):
        '''
        for BIES schema
        '''
        strans = [-float('inf')] * (len(self.LABEL.vocab)+1)
        # x:from y:to
        trans = [[-float('inf')] * (len(self.LABEL.vocab)+1) for _ in range((len(self.LABEL.vocab)+1))]
        B_idxs = []
        I_idxs = []
        E_idxs = []
        S_idxs = []
        permit_trans = set()
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('I-')):
                # strans[i] = -float('inf')  # cannot start with I-
                I_idxs.append(i)
                permit_trans.add((i, i))
                role = label[2:]
                corres_e_label = 'E-'+role
                corres_e_idx = self.LABEL.vocab.stoi[corres_e_label]
                permit_trans.add((i, corres_e_idx))
            elif label.startswith('B-'):
                strans[i] = 0
                B_idxs.append(i)
                role = label[2:]
                corres_i_label = 'I-'+role
                corres_i_idx = self.LABEL.vocab.stoi[corres_i_label]
                permit_trans.add((i, corres_i_idx))
                corres_e_label = 'E-'+role
                corres_e_idx = self.LABEL.vocab.stoi[corres_e_label]
                permit_trans.add((i, corres_e_idx))
            elif label.startswith('E-'):
                # strans[i] = -float('inf')
                E_idxs.append(i)
            elif label.startswith('S-'):
                strans[i] = 0
                S_idxs.append(i)
            elif label == '[prd]':
                pass
        # can strat with 'O'
        strans[-1] = 0
        # label
        for x, y in permit_trans:
            trans[x][y] = 0
        
        # start from E-
        for i in E_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0
        
        # start from S-
        for i in S_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0
        
        # start from 'O'
        for j in B_idxs:
            trans[-1][j] = 0
        for j in S_idxs:
            trans[-1][j] = 0
        trans[-1][-1] = 0

        return torch.tensor(strans), torch.tensor(trans), B_idxs, I_idxs, E_idxs, S_idxs, self.LABEL.vocab.stoi['[prd]']
        
    def prepare_viterbi_BES(self):
        '''
        for BES schema
        '''
        # -2 is 'I' and -1 is "O"
        strans = [-float('inf')] * (len(self.LABEL.vocab)+2)
        trans = [[-float('inf')] * (len(self.LABEL.vocab)+2) for _ in range((len(self.LABEL.vocab)+2))]
        B_idxs = []
        E_idxs = []
        S_idxs = []
        permit_trans = set()
        pair_dict = {}
        for i, label in enumerate(self.LABEL.vocab.itos):
            if(label.startswith('E-')):
                # strans[i] = -float('inf')  # cannot start with E-
                E_idxs.append(i)
            elif label.startswith('B-'):
                strans[i] = 0
                B_idxs.append(i)
                role = label[2:]
                corres_e_label = 'E-'+role
                corres_e_idx = self.LABEL.vocab.stoi[corres_e_label]
                permit_trans.add((i, corres_e_idx))
            elif label.startswith('S-'):
                strans[i] = 0
                S_idxs.append(i)
            elif label == '[prd]':
                pass
        # can strat with 'O'
        strans[-1] = 0
        # label
        for x, y in permit_trans:
            trans[x][y] = 0
        # start from B-
        for i in B_idxs:
            trans[i][-2] = 0

            # for j in E_idxs:
            #     trans[i][j] = 0

        # start from E-
        for i in E_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0
        
        # start from S-
        for i in S_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0
        
        # start from I
        for j in E_idxs:
            trans[-2][j] = 0
        trans[-2][-2] = 0

        # start from O
        for j in B_idxs:
            trans[-1][j] = 0
        for j in S_idxs:
            trans[-1][j] = 0
        trans[-1][-1] = 0

        for x, y in permit_trans:
            pair_dict[x] = y
            pair_dict[y] = x

        return torch.tensor(strans), torch.tensor(trans), B_idxs, E_idxs, S_idxs, pair_dict, self.LABEL.vocab.stoi['[prd]']

            
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
        # interpolation = args.itp
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT, ELMO = None, None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'elmo' in args.feat:
            ELMO = ElmoField('elmo')
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
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT),
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
        # if(args.use_pred):
        #     LABEL.vocab.extend(['Other'])
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
            'interpolation': args.itp,
            'encoder': args.encoder,
            'elmo_dropout': args.elmo_dropout
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")
        if (args.encoder != 'transformer'):
            optimizer = Adam(model.parameters(), **optimizer_args)
            scheduler = ExponentialLR(optimizer, **scheduler_args)
        else:
            optimizer = Adam(model.parameters(),
                             lr=0.04,
                             betas=(0.9, 0.98),
                             eps=1e-12)
            scheduler = VLR(optimizer, warmup_steps=8000)

        return cls(args, model, transform, optimizer, scheduler)
