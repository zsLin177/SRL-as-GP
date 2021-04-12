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
    'biaffine-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.biaffine.dep.lstm.char',
    'biaffine-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.biaffine.dep.lstm.char',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.dep.lstm.char',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.dep.lstm.char',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf2o.dep.lstm.char',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf2o.dep.lstm.char',
    'biaffine-dep-roberta-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.biaffine.dep.roberta',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.biaffine.dep.electra',
    'biaffine-dep-xlmr': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ud.biaffine.dep.xlmr',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.con.lstm.char',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.con.lstm.char',
    'crf-con-roberta-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.con.roberta',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.con.electra',
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/dm.biaffine.sdp.lstm.tag-char-lemma',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/dm.vi.sdp.lstm.tag-char-lemma'
}

CONFIG = {
    'biaffine-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.biaffine.dep.lstm.char.ini',
    'biaffine-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.biaffine.dep.lstm.char.ini',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.dep.lstm.char.ini',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.dep.lstm.char.ini',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf2o.dep.lstm.char.ini',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf2o.dep.lstm.char.ini',
    'biaffine-dep-roberta-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.biaffine.dep.roberta.ini',
    'biaffine-dep-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.biaffine.dep.electra.ini',
    'biaffine-dep-xlmr': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ud.biaffine.dep.xlmr.ini',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.con.lstm.char.ini',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.con.lstm.char.ini',
    'crf-con-roberta-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ptb.crf.con.roberta.ini',
    'crf-con-electra-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/ctb7.crf.con.electra.ini',
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/dm.biaffine.sdp.lstm.tag-char-lemma.ini',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/v1.1.0/dm.vi.sdp.lstm.tag-char-lemm.ini'
}
