# -*- coding: utf-8 -*-
import pdb
import os
from datetime import datetime, timedelta
import subprocess
import dill
import supar
import torch
import torch.distributed as dist
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.logging import init_logger, logger
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
import random


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
        elif(rel == ['[prd]']):
            column.append('*')
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



def get_results(gold_path, pred_path, file_seed):
    _SRL_CONLL_EVAL_SCRIPT = 'conll05-original-style/eval.sh'
    tgt_temp_file = 'tgt_temp_file' + file_seed
    change2(pred_path, tgt_temp_file)
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]

    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, tgt_temp_file, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    # pdb.set_trace()
    # temp = str(eval_info).strip().split("\\n")
    conll_recall = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_precision = float(str(eval_info2).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[5])

    return conll_f1, lisa_f1


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform, optimizer=None, scheduler=None):
        self.args = args
        self.model = model
        self.transform = transform
        self.optimizer = optimizer
        self.scheduler = scheduler

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
              verbose=True,
              **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size // args.update_steps, args.buckets, True,
                    dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(
            f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        # logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n")

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()
        # rand_file_seed1 = random.randint(1,100)
        # rand_file_seed2 = random.randint(1,100)
        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            # loss, dev_metric = self._evaluate(dev.loader)
            dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':5} - {dev_metric}")
            # dev_conll_f1, dev_lisa_f1, test_conll_f1, test_lisa_f1 = 0, 0, 0, 0
            # if(dev_pred != None):
            #     dev_ds = self.predict(args.dev, args.dev_pred)
            #     # pdb.set_trace()
            #     dev_conll_f1, dev_lisa_f1 = get_results(args.dev_gold, args.dev_pred, str(rand_file_seed1)+'-'+str(rand_file_seed2))    
            # logger.info(f"dev_conllf1:{dev_conll_f1:6.4} - dev_lisaf1:{dev_lisa_f1:6.4}")

            # loss, test_metric = self._evaluate(test.loader)
            # logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

            test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':5} - {test_metric}")
            # if(test_pred != None):
            #     test_ds = self.predict(args.test, args.test_pred)
            #     test_conll_f1, test_lisa_f1 = get_results(args.test_gold, args.test_pred, str(rand_file_seed1)+'-'+str(rand_file_seed2))
            # logger.info(f"test_conllf1:{test_conll_f1:6.4} - test_lisaf1:{test_lisa_f1:6.4}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        # loss, metric = self.load(**args)._evaluate(test.loader)
        metric = self.load(**args)._evaluate(test.loader)
        # metric = self.load(**args)._evaluate(dev.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':5} {best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        # loss, metric = self._evaluate(dataset.loader)
        metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        # logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"- {metric}")
        logger.info(
            f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s"
        )

        # return loss, metric
        return metric

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                prob=False,
                conll05=False,
                **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")

        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(
            f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s"
        )
        if(args.conll05):
            rand_file_seed1 = random.randint(1,100)
            rand_file_seed2 = random.randint(1,100)
            test_conll_f1, test_lisa_f1 = 0, 0
            test_conll_f1, test_lisa_f1 = get_results(args.gold, pred, str(rand_file_seed1)+'-'+str(rand_file_seed2))
            logger.info(f"test_conllf1:{test_conll_f1:6.4} - test_lisaf1:{test_lisa_f1:6.4}")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained parser defined in ``supar.PRETRAINED``
                  to load from cache or download, e.g., ``'crf-dep-en'``.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations and initiate the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dependency.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(path):
            state = torch.load(path)
        else:
            state = torch.hub.load_state_dict_from_url(
                supar.PRETRAINED[path] if path in supar.PRETRAINED else path)
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {
            'name': self.NAME,
            'args': args,
            'state_dict': state_dict,
            'pretrained': pretrained,
            'transform': self.transform
        }
        torch.save(state, path, pickle_module=dill)
