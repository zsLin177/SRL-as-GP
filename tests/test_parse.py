# -*- coding: utf-8 -*-

import os

import supar
import torch
from supar import Parser


def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    for n, m in supar.NAME.items():
        parser = Parser.load(n)
        parser.predict(sentence, prob=True)
        parser.predict(' '.join(sentence), prob=True, lang=('zh' if n.endswith('zh') else 'en'))
        os.remove(os.path.join(torch.hub.get_dir(), 'checkpoints', m))
