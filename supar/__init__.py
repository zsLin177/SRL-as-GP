# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser,
                      BiaffineSemanticDependencyParser, CRF2oDependencyParser,
                      CRFConstituencyParser, CRFDependencyParser,
                      CRFNPDependencyParser, Parser, VIConstituencyParser,
                      VIDependencyParser, VISemanticDependencyParser)

__all__ = ['BiaffineDependencyParser',
           'CRFNPDependencyParser',
           'CRFDependencyParser',
           'CRF2oDependencyParser',
           'VIDependencyParser',
           'CRFConstituencyParser',
           'VIConstituencyParser',
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
                                             VIConstituencyParser,
                                             BiaffineSemanticDependencyParser,
                                             VISemanticDependencyParser]}

MODEL = {
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.zip'
}

CONFIG = {
    'biaffine-sdp-en': 'http://hlt.suda.edu.cn/LA/yzhang/supar/dm.biaffine.sdp.tag-char-lemma.ini'
}
