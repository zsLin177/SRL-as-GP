# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser,
                      BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      CRFNPDependencyParser, Parser, VIDependencyParser,
                      VISemanticDependencyParser)

__all__ = ['BiaffineDependencyParser',
           'CRFNPDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'VIDependencyParser',
           'CRFConstituencyParser',
           'BiaffineSemanticDependencyParser',
           'VISemanticDependencyParser',
           'Parser']

__version__ = '1.0.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser,
                                             CRFNPDependencyParser,
                                             CRFDependencyParser,
                                             CRF2oDependencyParser,
                                             VIDependencyParser,
                                             CRFConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

MODEL = {
    'biaffine-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.char.zip',
    'biaffine-dep-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.bert.zip',
    'biaffine-dep-bert-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.bert.zip',
    'crfnp-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crfnp.dependency.char.zip',
    'crfnp-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crfnp.dependency.char.zip',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf2o.dependency.char.zip',
    'vi-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.vi.dependency.char.zip',
    'vi-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.vi.dependency.char.zip',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.char.zip',
    'crf-con-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.bert.zip',
    'crf-con-bert-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.bert.zip',
    'vi-sdp-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.semantic.dependency.-bert.zip',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.semantic.dependency.char.zip'
}

CONFIG = {
    'biaffine-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.char.ini',
    'biaffine-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.char.ini',
    'biaffine-dep-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.biaffine.dependency.bert.ini',
    'biaffine-dep-bert-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.biaffine.dependency.bert.ini',
    'crfnp-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crfnp.dependency.char.ini',
    'crfnp-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crfnp.dependency.char.ini',
    'crf-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.dependency.char.ini',
    'crf-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.dependency.char.ini',
    'crf2o-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf2o.dependency.char.ini',
    'crf2o-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf2o.dependency.char.ini',
    'vi-dep-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.vi.dependency.char.ini',
    'vi-dep-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.vi.dependency.char.ini',
    'crf-con-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.char.ini',
    'crf-con-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.char.ini',
    'crf-con-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ptb.crf.constituency.bert.ini',
    'crf-con-bert-zh': 'http://hlt.suda.edu.cn/LA/yzhang/supar/ctb7.crf.constituency.bert.ini',
    'vi-sdp-bert-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.semantic.dependency.bert.ini',
    'vi-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.vi.semantic.dependency.char.ini'
}
