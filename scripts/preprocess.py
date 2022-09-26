# change conll05 to conllu-like data


import argparse
import nltk
import re
import os
from nltk.stem import WordNetLemmatizer
from generate_data import prepare, write_BES_data
lemmatizer = WordNetLemmatizer()

regex = re.compile(r' +')


def test_to_BE_graph(file_name, out_file_name):
    with open(file_name, 'r') as f:
        line_lsts = [line.split() for line in f]
        # [line_lst1,line_lst2,...] line_lst:[word, if_prd, relation with prd1, relation with prd2, ...]
    sentence_lsts = []
    start, i = 0, 0
    for line_lst in line_lsts:
        if len(line_lst) <= 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1
    
    new_sentence_lsts = []
    for sentence_lst in sentence_lsts:
        words = [sentence_lst[i][0] for i in range(len(sentence_lst))]
        poss = [tup[1] for tup in nltk.pos_tag(words)]
        pred_lemmas = get_lemma(words, poss)
        raw_lemmas = [sentence_lst[i][1] for i in range(len(sentence_lst))]
        lemmas = []
        # 保留原来的谓词的lemma，不用预测的
        for i in range(len(sentence_lst)):
            if(raw_lemmas[i] != '-'):
                lemmas.append(raw_lemmas[i])
            else:
                lemmas.append(pred_lemmas[i])
        new_sentence_lst = []
        for num, line_lst in enumerate(sentence_lst, 1):
            new_line_lst = [str(num), line_lst[0], lemmas[num-1], '_', poss[num-1], '_', '_', '_']
            # 还剩下倒数第二列放边，最后一列放'_'
            new_sentence_lst.append(new_line_lst)
        
        prd_map = {}  # 1:23,2:27
        for num, line_lst in enumerate(sentence_lst, 1):
            if(line_lst[1] != '-'):
                prd_map[len(prd_map)+1] = num
        
        arc_lsts = [[] for i in range(len(sentence_lst))]
        for key, value in prd_map.items():
            prd_head_idx = value
            rela_col = [line_lst[1+key] for line_lst in sentence_lst]
            arcs = get_crosstag_arcs(rela_col, prd_head_idx)

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]

        for i in range(len(arc_lsts)):
            arc_values = []
            for arc in arc_lsts[i]:
                head_idx = arc[0]
                label = arc[1]
                arc_values.append(str(head_idx)+':'+label)
            if(len(arc_values) > 0):
                new_sentence_lst[i] += ['|'.join(arc_values), '_']
            else:
                new_sentence_lst[i] += ['_', '_']
        new_sentence_lsts.append(new_sentence_lst)

    with open(out_file_name, 'w') as f:
        for sentence_lst in new_sentence_lsts:
            for line_lst in sentence_lst:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')

def train_to_BII_graph(file_name, out_file_name):
    # also for dev
    with open(file_name, 'r') as f:
        line_lsts = [regex.sub('\t', line.strip()).split('\t') for line in f]
        # [line_lst1,line_lst2,...] line_lst:[word, pos, *, *, if_sense, if_prd, relation with prd1, relation with prd2, ...]
    sentence_lsts = []
    start, i = 0, 0
    for line_lst in line_lsts:
        if len(line_lst) <= 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence_lst in sentence_lsts:
        words = [sentence_lst[i][0] for i in range(len(sentence_lst))]
        poss = [tup[1] for tup in nltk.pos_tag(words)]
        pred_lemmas = get_lemma(words, poss)
        raw_lemmas = [sentence_lst[i][5] for i in range(len(sentence_lst))]
        lemmas = []
        # 保留原来的谓词的lemma，不用预测的
        for i in range(len(sentence_lst)):
            if(raw_lemmas[i] != '-'):
                lemmas.append(raw_lemmas[i])
            else:
                lemmas.append(pred_lemmas[i])
        new_sentence_lst = []
        for num, line_lst in enumerate(sentence_lst, 1):
            new_line_lst = [str(num), line_lst[0], lemmas[num-1], '_', poss[num-1], '_', '_', '_']
            # 还剩下倒数第二列放边，最后一列放'_'
            new_sentence_lst.append(new_line_lst)

        # num_prd = len(sentence_lst[0]) - 6
        prd_map = {}  # 1:23,2:27
        for num, line_lst in enumerate(sentence_lst, 1):
            if(line_lst[5] != '-'):
                prd_map[len(prd_map)+1] = num

        arc_lsts = [[] for i in range(len(sentence_lst))]
        for key, value in prd_map.items():
            prd_head_idx = value
            rela_col = [line_lst[5+key] for line_lst in sentence_lst]
            arcs = BII_graph(rela_col, prd_head_idx)

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]
        
        for i in range(len(arc_lsts)):
            arc_values = []
            for arc in arc_lsts[i]:
                head_idx = arc[0]
                label = arc[1]
                arc_values.append(str(head_idx)+':'+label)
            if(len(arc_values) > 0):
                new_sentence_lst[i] += ['|'.join(arc_values), '_']
            else:
                new_sentence_lst[i] += ['_', '_']

        new_sentence_lsts.append(new_sentence_lst)

    with open(out_file_name, 'w') as f:
        for sentence_lst in new_sentence_lsts:
            for line_lst in sentence_lst:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')

def test_to_BII_graph(file_name, out_file_name):
    with open(file_name, 'r') as f:
        line_lsts = [line.split() for line in f]
        # [line_lst1,line_lst2,...] line_lst:[word, if_prd, relation with prd1, relation with prd2, ...]
    sentence_lsts = []
    start, i = 0, 0
    for line_lst in line_lsts:
        if len(line_lst) <= 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1
    
    new_sentence_lsts = []
    for sentence_lst in sentence_lsts:
        words = [sentence_lst[i][0] for i in range(len(sentence_lst))]
        poss = [tup[1] for tup in nltk.pos_tag(words)]
        pred_lemmas = get_lemma(words, poss)
        raw_lemmas = [sentence_lst[i][1] for i in range(len(sentence_lst))]
        lemmas = []
        # 保留原来的谓词的lemma，不用预测的
        for i in range(len(sentence_lst)):
            if(raw_lemmas[i] != '-'):
                lemmas.append(raw_lemmas[i])
            else:
                lemmas.append(pred_lemmas[i])
        new_sentence_lst = []
        for num, line_lst in enumerate(sentence_lst, 1):
            new_line_lst = [str(num), line_lst[0], lemmas[num-1], '_', poss[num-1], '_', '_', '_']
            # 还剩下倒数第二列放边，最后一列放'_'
            new_sentence_lst.append(new_line_lst)
        
        prd_map = {}  # 1:23,2:27
        for num, line_lst in enumerate(sentence_lst, 1):
            if(line_lst[1] != '-'):
                prd_map[len(prd_map)+1] = num
        
        arc_lsts = [[] for i in range(len(sentence_lst))]
        for key, value in prd_map.items():
            prd_head_idx = value
            rela_col = [line_lst[1+key] for line_lst in sentence_lst]
            arcs = BII_graph(rela_col, prd_head_idx)

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]

        for i in range(len(arc_lsts)):
            arc_values = []
            for arc in arc_lsts[i]:
                head_idx = arc[0]
                label = arc[1]
                arc_values.append(str(head_idx)+':'+label)
            if(len(arc_values) > 0):
                new_sentence_lst[i] += ['|'.join(arc_values), '_']
            else:
                new_sentence_lst[i] += ['_', '_']
        new_sentence_lsts.append(new_sentence_lst)

    with open(out_file_name, 'w') as f:
        for sentence_lst in new_sentence_lsts:
            for line_lst in sentence_lst:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')
 
def get_crosstag_arcs(rela_col, prd_head_idx):
    '''
    BI
    '''
    arcs = [[] for i in range(len(rela_col))]
    span_start = -1
    span_label = ''
    
    arcs[prd_head_idx-1].append((0, '[prd]'))

    for idx, rela in enumerate(rela_col, 1):
        if((rela[0] == '(') and rela != '(V*)' and rela[-1] != ')' and rela != '(V*'):
            # like "(A0*"
            span_start = idx
            value = rela[1:-1]
            span_label = value
            arcs[idx-1].append((prd_head_idx, 'B-'+value))
        elif(rela == '(V*)'):
            # arcs[idx-1].append((0, '[prd]'))
            pass
        elif(rela == '(V*'):
            # arcs[idx-1].append((0, '[prd]'))
            span_start = idx
            value = rela[1:-1]
            span_label = value
        elif(rela == '*'):
            if(span_start != -1):
                # in span
                # arcs[idx-1].append((prd_head_idx, 'I-'+span_label))
                pass
            else:
                continue
        elif(rela[0] == '(' and rela[-1] == ')'):
            # "(A0*)"
            arcs[idx-1].append((prd_head_idx, 'B-'+rela[1:-2]))
        else:
            # '*)'
            if(span_label != 'V'):
                arcs[idx-1].append((prd_head_idx, 'I-'+span_label))
            span_start = -1
            span_label = ''
    return arcs
            
def BII_graph(rela_col, prd_head_idx):
    '''
    BII
    '''
    arcs = [[] for i in range(len(rela_col))]
    span_start = -1
    span_label = ''
    
    arcs[prd_head_idx-1].append((0, '[prd]'))

    for idx, rela in enumerate(rela_col, 1):
        if((rela[0] == '(') and rela != '(V*)' and rela[-1] != ')' and rela != '(V*'):
            # like "(A0*"
            span_start = idx
            value = rela[1:-1]
            span_label = value
            arcs[idx-1].append((prd_head_idx, 'B-'+value))
        elif(rela == '(V*)'):
            # arcs[idx-1].append((0, '[prd]'))
            pass
        elif(rela == '(V*'):
            # arcs[idx-1].append((0, '[prd]'))
            span_start = idx
            value = rela[1:-1]
            span_label = value
        elif(rela == '*'):
            if(span_start != -1 and span_label != 'V'):
                # in span
                arcs[idx-1].append((prd_head_idx, 'I-'+span_label))
                pass
            else:
                continue
        elif(rela[0] == '(' and rela[-1] == ')'):
            # "(A0*)"
            arcs[idx-1].append((prd_head_idx, 'B-'+rela[1:-2]))
        else:
            # '*)'
            if(span_label != 'V'):
                arcs[idx-1].append((prd_head_idx, 'I-'+span_label))
            span_start = -1
            span_label = ''
    return arcs

def train_to_simple_graph(file_name):
    false_num = 0
    with open(file_name, 'r') as f:
        line_lsts = [regex.sub('\t', line.strip()).split('\t') for line in f]
        # [line_lst1,line_lst2,...] line_lst:[word, pos, *, *, if_sense, if_prd, relation with prd1, relation with prd2, ...]
    sentence_lsts = []
    start, i = 0, 0
    for line_lst in line_lsts:
        if len(line_lst) <= 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1
    
    new_sentence_lsts = []
    for sentence_lst in sentence_lsts:
        words = [sentence_lst[i][0] for i in range(len(sentence_lst))]
        poss = [tup[1] for tup in nltk.pos_tag(words)]
        lemmas = get_lemma(words, poss)
        new_sentence_lst = []
        for num, line_lst in enumerate(sentence_lst, 1):
            new_line_lst = [num, line_lst[0], lemmas[num-1], '_', poss[num-1], '_', '_', '_']
            # 还剩下倒数第二列放边，最后一列放'_'
            new_sentence_lst.append(new_line_lst)
        
        num_prd = len(sentence_lst[0]) - 6
        prd_map = {}  # 1:23,2:27
        for num, line_lst in enumerate(sentence_lst, 1):
            if(line_lst[5] != '-'):
                prd_map[len(prd_map)+1] = num
        
        arc_lsts = [[] for i in range(len(sentence_lst))]
        if_overlap = [[0,0] for i in range(len(sentence_lst))]
        for key, value in prd_map.items():
            prd_head_idx = value
            # print(sentence_lst)
            # print(key, value)
            rela_col = [line_lst[5+key] for line_lst in sentence_lst]
            arcs, temp= get_crosstag_arcs(rela_col, prd_head_idx)
            for i in range(len(temp)):
                if(temp[i][0]):
                    if_overlap[i][0]=1
                if(temp[i][1]):
                    if_overlap[i][1]=1

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]
        for i in range(len(if_overlap)):
            if(if_overlap[i][0] and if_overlap[i][1]):
                false_num += 1
                print(sentence_lst[i])
                print(sentence_lst)
            
        # 一个span作为多个谓词的论元，则这个span只保存一个
        for i in range(len(arc_lsts)):
            arc_lsts[i] = list(set(arc_lsts[i]))
        
    print(false_num)
   
def get_lemma(words, poss):
    lemmas = []
    for word, tag in zip(words, poss):
        if tag.startswith('NN'):
            lemmas.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemmas.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemmas.append(lemmatizer.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            lemmas.append(lemmatizer.lemmatize(word, pos='r'))
        else:
            lemmas.append(word)
    return lemmas


if __name__ == '__main__':
    # file_name = 'test-brown'
    # out_file_name = 'sc_tmp_brown.conllu'
    # test_to_BE_graph(file_name, out_file_name)

    parser = argparse.ArgumentParser(
        description='Convert the file of prop format to conllu format in BES schema.'
    )
    parser.add_argument('--prop', help='path to the prop file')
    parser.add_argument('--file', help='path to the converted file')
    args = parser.parse_args()

    test_to_BE_graph(args.prop, args.file+'_tmp')
    gold_spans = prepare(args.file+'_tmp')
    write_BES_data(args.file+'_tmp', gold_spans, args.file)
    os.remove(args.file+'_tmp')



