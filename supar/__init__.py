# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser, BiaffineSemanticDependencyParser, CRF2oDependencyParser, CRFConstituencyParser,
                      CRFDependencyParser, Parser, VIConstituencyParser, VIDependencyParser, VISemanticDependencyParser)

__all__ = ['BiaffineDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'VIDependencyParser',
           'CRFConstituencyParser',
           'VIConstituencyParser',
           'BiaffineSemanticDependencyParser',
           'VISemanticDependencyParser',
           'Parser']

__version__ = '1.1.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             VIDependencyParser,
                                             CRFConstituencyParser,
                                             VIConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

MODEL = {
    'biaffine-dep-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.bert.zip',
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.zip',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.zip'
}

CONFIG = {
    'biaffine-dep-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.bert.ini',
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.ini',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.ini'
}
