# -*- coding: utf-8 -*-

import argparse

from supar import VISrlParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create SRL Parser using Variational Inference.')
    parser.set_defaults(Parser=VISrlParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', default='tag,char,lemma', help='additional features to use，separated by commas.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/sdp/DM/train.conllu', help='path to train file')
    subparser.add_argument('--dev', default='data/sdp/DM/dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default='data/sdp/DM/test.conllu', help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=300, type=int, help='dimension of embeddings')
    subparser.add_argument('--n_pretrained_embed',
                           default=300,
                           type=int,
                           help='dimension of pretrained embeddings')
    subparser.add_argument('--bert', default='bert-large-uncased', help='which bert model to use')
    subparser.add_argument('--itp', default=0.1, type=float, help='inter...')
    subparser.add_argument('--inference', default='mfvi', choices=['mfvi', 'lbp'], help='approximate inference methods')
    subparser.add_argument('--n_lstm_layers', default=3, type=int)
    subparser.add_argument('--encoder', default='lstm')
    subparser.add_argument('--clip', default=5.0, type=float)
    subparser.add_argument('--split',
                           action='store_true',
                           help='whether to use different mlp for predicate and arg')
    subparser.add_argument('--repr_gold',
                           action='store_true',
                           help='whether to use gold predicates during train to repr label')
    # subparser.add_argument('--dev_gold',
    #                        default='data/conll05-original-style/sc-dev.final')
    # subparser.add_argument('--dev_pred',
    #                        default='dev_mfvi_pred')
    # subparser.add_argument('--test_gold',
    #                        default='data/conll05-original-style/sc-wsj.final')
    # subparser.add_argument('--test_pred',
    #                        default='test_mfvi_pred')
    subparser.add_argument('--min_freq',
                           default=7,
                           type=int,
                           help='The minimum frequency needed to include a token in the vocabulary')
    subparser.add_argument('--train_given_prd',
                           action='store_true',
                           default=False,
                           help='whether use predicate embedding')

    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    # predict
    subparser = subparsers.add_parser(
        'predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob',
                           action='store_true',
                           help='whether to output probs')
    subparser.add_argument('--buckets',
                           default=8,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='data/sdp/DM/test.conllu',
                           help='path to dataset')
    subparser.add_argument('--pred',
                           default='pred.conllu',
                           help='path to predicted result')
    subparser.add_argument('--task', choices=['05', '09', '12'], required=True, help='which dataset')
    subparser.add_argument('--gold',
                           default='data/conll05-original-style/sc-wsj.final')
    subparser.add_argument('--vtb',
                           action='store_true',
                           default=False,
                           help='whether to use viterbi')
    subparser.add_argument('--given_prd',
                           action='store_true',
                           default=False,
                           help='whether given prdicates')
    parse(parser)


if __name__ == "__main__":
    main()
