# -*- coding: utf-8 -*-

import os

import supar
import torch
from supar import Parser


def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    for name in supar.MODEL:
        parser = Parser.load(name, device='cuda:0')
        parser.predict(sentence, prob=True)
        parser.predict(' '.join(sentence), prob=True, lang=('zh' if name.endswith('zh') else 'en'))
        os.remove(os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(supar.MODEL[name])))
