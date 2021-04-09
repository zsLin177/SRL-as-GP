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
    'biaffine-dep-lstm-char': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.lstm.char',
    'biaffine-dep-lstm-char-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.lstm.char',
    'crf-con-lstm-char': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.lstm.char',
    'crf-con-lstm-char-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.lstm.char',
    'biaffine-dep-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.roberta',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.electra',
    'crf-con-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.roberta',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.electra',
    'biaffine-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.lstm.tag-char-lemma',
    'vi-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.sdp.lstm.tag-char-lemma'
}

CONFIG = {
    'biaffine-dep-lstm-char': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.lstm.char.ini',
    'biaffine-dep-lstm-char-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.lstm.char.ini',
    'crf-con-lstm-char': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.lstm.char.ini',
    'crf-con-lstm-char-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.lstm.char.ini',
    'biaffine-dep-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dep.roberta.ini',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dep.electra.ini',
    'crf-con-roberta': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.con.roberta.ini',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.con.electra.ini',
    'biaffine-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.lstm.tag-char-lemma.ini',
    'vi-sdp': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.sdp.lstm.tag-char-lemma.ini'
}
