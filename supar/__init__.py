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
    'biaffine-dep-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.roberta.zip',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.electra.zip',
    'crf-con-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.roberta.zip',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.electra.zip',
    'biaffine-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.lstm.tag-char-lemma.zip',
    'vi-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.sdp.lstm.tag-char-lemma.zip'
}

CONFIG = {
    'biaffine-dep-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.roberta.ini',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.electra.ini',
    'crf-con-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.roberta.ini',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.electra.ini',
    'biaffine-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.lstm.tag-char-lemma.ini',
    'vi-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.sdp.lstm.tag-char-lemma.ini'
}
