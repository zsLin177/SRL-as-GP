import os
import pdb
import subprocess
import torch
import torch.nn as nn
from supar.models import (BiaffineSrlModel, BiaffineSpanSrlModel,
                          VISrlModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField, SpanSrlFiled
from supar.utils.logging import get_logger, progress_bar
from supar.utils.logging import init_logger, logger
from supar.utils.metric import ChartMetric, SrlMetric
from supar.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler

logger = get_logger(__name__)

def change2(source_file, tgt_file):
    # change simple crosstag conllu to target type
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # 先找出所有的谓词
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[2])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column = produce_column_3(this_prd_arc, this_prd_idx)
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)

    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')


def produce_column_3(relas, prd_idx):
    # used for simple crosstag
    # 暂时是直接按照预测的B、I进行划分
    column = []
    # span_start = -1
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        # print(i)
        # print(relas)
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            column.append('(V*)')
            i += 1
        elif (len(rel) == 0):
            column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if (position_tag in ('B', 'I')):
                # 这里把I也考虑进来，防止第一个是I（I之前没有B，那么这个I当成B）
                span_start = i
                span_end = -1
                i += 1
                # labels = {}
                # labels[label] = 1
                while (i < len(relas)):
                    if (len(relas[i]) == 0):
                        i += 1
                        continue
                    else:
                        # relas[i][0][0] == 'B' or 'I'
                        if (relas[i][0][0] == 'B'):
                            break
                        else:
                            span_end = i
                            label2 = relas[i][0][2:]  # 以后面那个作为label
                            i += 1
                            break
                if (span_end != -1):
                    if (label == label2):
                        length = span_end - span_start + 1
                        column.append('(' + label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
                    else:
                        length = span_end - span_start + 1
                        column += ['*'] * length
                else:
                    column.append('(' + label + '*' + ')')
                    column += ['*'] * (i - 1 - span_start)
    return column



def get_results(gold_path, pred_path):
    _SRL_CONLL_EVAL_SCRIPT = 'conll05-original-style/eval.sh'
    tgt_temp_file = 'tgt_temp_file'
    change2(pred_path, tgt_temp_file)
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]

    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, pred_path, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    # pdb.set_trace()
    # temp = str(eval_info).strip().split("\\n")
    conll_recall = float(str(eval_info).strip().split("\\n")[-42:][6].strip().split()[5])
    conll_precision = float(str(eval_info2).strip().split("\\n")[-42:][6].strip().split()[5])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("\\n")[-42:][6].strip().split()[6])

    return conll_f1, lisa_f1


class VLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=8000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(VLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        scale = min(pow(epoch, -0.5), epoch * pow(self.warmup_steps, -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class BiaffineSpanSrlParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 20178.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    """

    NAME = 'biaffine-span_based-semantic-role-labeling'
    MODEL = BiaffineSpanSrlModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.LABEL, self.SPAN = self.transform.PHEAD

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

        for words, *feats, edges, labels, spans in bar:
            self.optimizer.zero_grad()
            # pdb.set_trace()
            mask1 = words.ne(self.WORD.pad_index)
            mask = mask1.unsqueeze(1) & mask1.unsqueeze(2)
            mask[:, 0] = 0
            pred_mask = mask1 & edges[..., 0].eq(1)
            pred_mask[:, 0] = 0
            s_edge, s_label, encoder_out = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            span_loss = self.model.span_loss(pred_mask, mask, spans, encoder_out)
            loss += span_loss
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
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        charts, probs = [], []
        for words, *feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
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
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs
        # pdb.set_trace()

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
        LABEL = ChartField('labels', fn=CoNLL.get_BI_labels)
        SPAN = SpanSrlFiled('spans', build_fn=CoNLL.get_span_labels, fn=CoNLL.get_spans)
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=(EDGE, LABEL, SPAN))

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
        SPAN.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_span_labels': len(SPAN.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'interpolation': interpolation,
            'encoder': args.encoder
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

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.LABEL = self.transform.PHEAD

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

        for words, *feats, edges, labels in bar:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            if(self.args.repr_gold):
                s_edge, s_label = self.model(words, feats, edges)
            else:
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
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        charts, probs = [], []
        for words, *feats in progress_bar(loader):
            # pdb.set_trace()
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
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
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs
        # pdb.set_trace()

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
        if(args.use_pred):
            LABEL.vocab.extend(['Other'])
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
            'encoder': args.encoder
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
    def _predict(self, loader):

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
