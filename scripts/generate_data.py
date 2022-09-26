import torch.nn.functional as F
import pdb
from pylab import *
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import collections

role_set = ['A0', 'A1', 'AM-TMP', 'A2', 'AM-MOD', 'AM-ADV', 'AM-LOC']
# sec_o_p  = [89.85, 83.62, 83.7, 75.6, 96.6, 70.6, 72.31]
# sec_o_r  = [91.89, 85.35, 84.54, 74.23, 98.0, 60.28, 64.74]

# fir_o_p  = [89.46, 82.92, 83.85, 75.07, 96.59, 69.68, 70.54]
# fir_o_r  = [91.05, 84.51, 84.08, 73.51, 97.82, 59.49, 65.29]

# for i in range(7):
#     name = role_set[i]
#     fir_p = fir_o_p[i]
#     fir_r = fir_o_r[i]
#     fir_f1 = 2*fir_p*fir_r/(fir_p+fir_r)

#     sec_p = sec_o_p[i]
#     sec_r = sec_o_r[i]
#     sec_f1 = 2*sec_p*sec_r/(sec_p+sec_r)

#     delta_f1 = sec_f1 - fir_f1
#     print(name+': ', '%.2f'%fir_f1, '%.2f'%sec_f1, '%.2f'%delta_f1, end=' ')
#     print()
    
# 统计 ['A0', 'A1', 'AM-TMP', 'A2', 'AM-MOD', 'AM-ADV', 'AM-LOC']占有二阶结构的比例
# file_name = 'sc-conll5-wsj.conllu'
# num_sum = [0] * 7
# contain_2o = [0] * 7
# with open(file_name, 'r') as f:
#         lines = [line.strip() for line in f]
#         for line in lines:
#             if(len(line)>1):
#                 s = line.split('\t')[8]
#                 if(s != '_'):
#                     args = [tmp_s.split(':')[1] for tmp_s in s.split('|')]
#                     flag = 0
#                     if(len(args)>1):
#                         flag = 1
#                     for arg in args:
#                         for i in range(7):
#                             if(role_set[i] in arg):
#                                 num_sum[i] += 1
#                                 if(flag == 1):
#                                     contain_2o[i] += 1

# ratio = [contain_2o[i]/num_sum[i] for i in range(7)]
# for i in range(7):
#     print(role_set[i], '%.4f'%ratio[i])


def change2(source_file, gold_file, task):
    '''
    for BE
    '''
    sum_conf1_count = 0
    sum_conf2_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    # change simple crosstag conllu to target type
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # 先找出所有的谓词
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2 = produce_column_3(this_prd_arc, this_prd_idx)
            sum_conf1_count += con1
            sum_conf2_count += con2
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_conf1_count))
    print('conflict 2:'+str(sum_conf2_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')


def produce_column_3(relas, prd_idx):
    # used for simple crosstag
    # 暂时是直接按照预测的B、I进行划分
    count = 0
    count2 = 0
    column = []
    # all spans: [(prd_idx, start_idx, end_idx, label),...,]
    spans = []
    
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        # print(i)
        # print(relas)
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            column.append('*')
            i += 1
        elif(rel == ['Other']):
            column.append('*')
            i += 1
        elif (len(rel) == 0):
            column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if(position_tag == 'I'):
                if(i!=len(relas)-1 and len(relas[i+1])>0 and relas[i+1][0][0]=='I'):
                    column.append('('+ label + '*')
                    column.append('*' + ')')
                    i += 2
                else:
                    # column.append('(' + label + '*' + ')') # turn into b
                    # spans.append((prd_idx, i, i, label))

                    column.append('*')   # 直接把冲突的I删掉
                    i += 1
                    count += 1
            else:
            # if(position_tag in ('B', 'I')):
                span_start = i
                span_end = -1
                i += 1
                # labels = {}
                # labels[label] = 1
                while (i < len(relas)):
                    if (len(relas[i]) == 0 or relas[i] == ['Other']):
                        i += 1
                        continue
                    else:
                        # relas[i][0][0] == 'B' or 'I'
                        if (relas[i][0][0] == 'B'):
                            break
                        else:
                            span_end = i
                            label2 = relas[i][0][2:]  # 以后面那个作为label
                            i += 1
                            break
                if (span_end != -1):
                    if (label == label2):
                        length = span_end - span_start + 1
                        column.append('(' + label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
                        spans.append((prd_idx, span_start+1, span_end+1, label))
                    else:
                        length = span_end - span_start + 1
                        column += ['*'] * length
                        count2 += 1
                else:
                    column.append('(' + label + '*' + ')')
                    column += ['*'] * (i - 1 - span_start)
                    spans.append((prd_idx, span_start+1, span_start+1, label))
    return column, count, count2, spans

def prepare(source_file):
    # source_file must be the format conllu
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1
    # [[(),(),()],[]]
    idx = 0
    all_sentence_spans = []
    for sentence in sentences:
        idx += 1
        # if idx == 31:
        #     pdb.set_trace()
        this_sents_spans = []
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]
        # 先找出所有的谓词
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2, spans = produce_column_3(this_prd_arc, this_prd_idx)
            this_sents_spans.extend(spans)
            # sum_conf1_count += con1
            # sum_conf2_count += con2
            # new_columns.append(this_column)
        all_sentence_spans.append(this_sents_spans)
    return all_sentence_spans


def prepare_1(source_file):
    with open(source_file, 'r') as f:
        line_lsts = [line.split() for line in f]
        # [line_lst1,line_lst2,...] line_lst:[word, relation with prd1, relation with prd2, ...]
        
    sentence_lsts = []
    start, i = 0, 0
    for line_lst in line_lsts:
        if len(line_lst) < 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1
    
    spans = []
    for sens_lst in sentence_lsts:
        if(len(sens_lst[0]) == 1):
            spans.append([])
            continue
        this_sents_spans = []
        predicate_idx2trueid = {} # {1:12, 2:23}
        for true_idx, line_lst in enumerate(sens_lst, 1):
            if(line_lst[0] != '-'):
                predicate_idx2trueid[len(predicate_idx2trueid)+1] = true_idx
        
        for i in range(len(predicate_idx2trueid)):
            prd_head_idx = predicate_idx2trueid[i+1]
            this_prd_relas = [line_lst[i+1] for line_lst in sens_lst]
            this_sents_prd_spans = get_spans_from_final(this_prd_relas, prd_head_idx)
            this_sents_spans.extend(this_sents_prd_spans)
        spans.append(this_sents_spans)
    
    return spans

def get_spans_from_final(rela_col, prd_head_idx):
    spans = []
    span_start = -1
    span_label = ''
    for idx, rela in enumerate(rela_col, 1):
        if((rela[0] == '(') and rela != '(V*)' and rela[-1] != ')' and rela != '(V*'):
            # like "(A0*"
            span_start = idx
            value = rela[1:-1]
            span_label = value
        elif(rela == '(V*)'):
            # arcs[idx-1].append((0, '[prd]'))
            continue
        elif(rela == '(V*'):
            # arcs[idx-1].append((0, '[prd]'))
            span_start = idx
            value = rela[1:-1]
            span_label = value
        elif(rela == '*'):
            if(span_start != -1):
                # in span
                # arcs[idx-1].append((prd_head_idx, 'I-'+span_label))  # for biii
                pass
            else:
                continue
        elif(rela[0] == '(' and rela[-1] == ')'):
            # "(A0*)"
            # arcs[idx-1].append((prd_head_idx, 'B-'+rela[1:-2]))
            spans.append((prd_head_idx, idx, idx, rela[1:-2]))
        else:
            # '*)'
            if(span_label != 'V'):
                # arcs[idx-1].append((prd_head_idx, 'I-'+span_label))
                spans.append((prd_head_idx, span_start, idx, span_label))
            span_start = -1
            span_label = ''
    return spans


def analy(pred_spans, gold_spans):
    pred_one_word_arg = 0
    pred_corr_one_word_arg = 0
    gold_one_word_arg = 0

    pred_multi_word_arg = 0
    pred_corr_multi_word_arg = 0
    gold_multi_word_arg = 0

    pred_all_arg = 0
    pred_all_corr_arg = 0
    gold_all_arg = 0

    for i in range(len(pred_spans)):
        pred_s = set(pred_spans[i])
        gold_s = set(gold_spans[i])

        pred_all_arg += len(pred_s)
        gold_all_arg += len(gold_s)
        pred_all_corr_arg += len(pred_s & gold_s)

        pred_one_s = set([tup for tup in pred_s if tup[1]==tup[2]])
        pred_mul_s = pred_s - pred_one_s
        gold_one_s = set([tup for tup in gold_s if tup[1]==tup[2]])
        gold_mul_s = gold_s - gold_one_s

        pred_one_word_arg += len(pred_one_s)
        gold_one_word_arg += len(gold_one_s)
        pred_corr_one_word_arg += len(pred_one_s & gold_one_s)

        pred_multi_word_arg += len(pred_mul_s)
        gold_multi_word_arg += len(gold_mul_s)
        pred_corr_multi_word_arg += len(pred_mul_s & gold_mul_s)
    all_p = pred_all_corr_arg/pred_all_arg
    all_r = pred_all_corr_arg/gold_all_arg
    all_f = 2*all_p*all_r/(all_r+all_p)

    one_p = pred_corr_one_word_arg/pred_one_word_arg
    one_r = pred_corr_one_word_arg/gold_one_word_arg
    one_f = 2*one_p*one_r/(one_r+one_p)

    mul_p = pred_corr_multi_word_arg/pred_multi_word_arg
    mul_r = pred_corr_multi_word_arg/gold_multi_word_arg
    mul_f = 2*mul_p*mul_r/(mul_p+mul_r)

    print('all:P,R,F:', '%.2f %.2f %.2f'%(all_p*100, all_r*100, all_f*100))
    print('one:P,R,F:', '%.2f %.2f %.2f'%(one_p*100, one_r*100, one_f*100))
    print('mul:P,R,F:', '%.2f %.2f %.2f'%(mul_p*100, mul_r*100, mul_f*100))
    # pdb.set_trace()


def detailed_analy(pred_spans, gold_spans, gate_size):
    length_range = [i for i in range(1, gate_size+2)]
    F_values = []
    P_values = []
    R_values = []
    # pred_sized_spans[0] = [] = gold_sized_spans[0]
    pred_sized_spans = [[] for i in range(gate_size+2)]
    gold_sized_spans = [[] for i in range(gate_size+2)]

    sum_arg_gold = 0
    sized_arg_gold = [0] * (gate_size+2)
    
    for i in range(len(pred_spans)):
        pred_s = set(pred_spans[i])
        gold_s = set(gold_spans[i])

        pred_tmp_lsts = [[] for i in range(gate_size+2)]
        for j, span in enumerate(pred_s):
            span_len = span[2] - span[1] + 1
            sum_arg_gold += 1
            if(span_len<=gate_size):
                pred_tmp_lsts[span_len].append(span)
                sized_arg_gold[span_len] += 1
            else:
                pred_tmp_lsts[-1].append(span)
                sized_arg_gold[-1] += 1
        for j in range(1, gate_size+2):
            pred_sized_spans[j].append(pred_tmp_lsts[j])
        
        gold_tmp_lsts = [[] for i in range(gate_size+2)]
        for j, span in enumerate(gold_s):
            span_len = span[2] - span[1] + 1
            if(span_len<=gate_size):
                gold_tmp_lsts[span_len].append(span)
            else:
                gold_tmp_lsts[-1].append(span)
        for j in range(1, gate_size+2):
            gold_sized_spans[j].append(gold_tmp_lsts[j])
    
    for i in range(1, gate_size+2):
        p, r, f = analy_all(pred_sized_spans[i], gold_sized_spans[i], i)
        F_values.append(round(f*100,2))
        P_values.append(round(p*100,2))
        R_values.append(round(r*100,2))


    # pdb.set_trace()
    # for i in range(1, gate_size+2):
    #     print(i, 'ratio:', '%.2f'%(sized_arg_gold[i]/sum_arg_gold*100), end='  ')
    #     print()

    return (length_range, P_values, R_values, F_values)
    

def span_analy(pred_spans, gold_spans, span_ranges):
    x = [i+1 for i in range(len(span_ranges))]
    F_values = []
    P_values = []
    R_values = []
    # pred_sized_spans[0] = [] = gold_sized_spans[0]
    pred_sized_spans = [[] for i in range(len(span_ranges)+1)]
    gold_sized_spans = [[] for i in range(len(span_ranges)+1)]

    for i in range(len(pred_spans)):
        pred_s = set(pred_spans[i])
        gold_s = set(gold_spans[i])
        pred_tmp_lsts = [[] for i in range(len(span_ranges)+1)]
        for j, span in enumerate(pred_s):
            span_len = span[2] - span[1] + 1
            idx = find_inwhich_span(span_len, span_ranges, 0, len(span_ranges)-1) + 1
            pred_tmp_lsts[idx].append(span)
        for j in range(1, (len(span_ranges)+1)):
            pred_sized_spans[j].append(pred_tmp_lsts[j])
        
        gold_tmp_lsts = [[] for i in range(len(span_ranges)+1)]
        for j, span in enumerate(gold_s):
            span_len = span[2] - span[1] + 1
            idx = find_inwhich_span(span_len, span_ranges, 0, len(span_ranges)-1) + 1
            gold_tmp_lsts[idx].append(span)
        for j in range(1, (len(span_ranges)+1)):
            gold_sized_spans[j].append(gold_tmp_lsts[j])
    
    for i in range(1, (len(span_ranges)+1)):
        p, r, f = analy_all(pred_sized_spans[i], gold_sized_spans[i], i)
        F_values.append(round(f*100,2))
        P_values.append(round(p*100,2))
        R_values.append(round(r*100,2))

    return (x, P_values, R_values, F_values)




def draw_pic(our2o_x, our2o_y, he_x, he_y, our1o_x, our1o_y, name, x_labels=None, name1=None, name2=None):
    # pdb.set_trace()
    plt.cla()
    plt.ylim(77, 91)
    plt.plot(our2o_x, our2o_y, 'o-', color='#FFB344', alpha=0.8, linewidth=1, label=name1)
    plt.plot(our1o_x, our1o_y, 's-', color='#E05D5D', alpha=0.8, linewidth=1, label=name2)
    plt.plot(he_x, he_y, '^-', color='#22577A', alpha=0.8, linewidth=1, label='He et al.(2018)')
    plt.tick_params(labelsize=12)
    if(x_labels != None):
        plt.xticks(our2o_x, x_labels, fontsize=13)
    plt.legend(loc='best')
    plt.xlabel('width (proportion)', fontsize=15)
    # plt.ylabel(name)
    # plt.show()
    plt.savefig(name+'.jpg')


def draw_all_pic(he, our2o, our1o, x_labels=None, name1=None, name2=None):
    draw_pic(our2o[0], our2o[1], he[0], he[1], our1o[0], our1o[1], 'P', x_labels, name1, name2)
    draw_pic(our2o[0], our2o[2], he[0], he[2], our1o[0], our1o[2], 'R', x_labels, name1, name2)
    draw_pic(our2o[0], our2o[3], he[0], he[3], our1o[0], our1o[3], 'F1', x_labels, name1, name2)

def analy_all(pred_spans, gold_spans, span_len):
    pred_all_arg = 0
    pred_all_corr_arg = 0
    gold_all_arg = 0

    for i in range(len(pred_spans)):
        pred_s = set(pred_spans[i])
        gold_s = set(gold_spans[i])

        pred_all_arg += len(pred_s)
        gold_all_arg += len(gold_s)
        pred_all_corr_arg += len(pred_s & gold_s)

    all_p = pred_all_corr_arg/pred_all_arg
    all_r = pred_all_corr_arg/gold_all_arg
    all_f = 2*all_p*all_r/(all_r+all_p)

    print('Results of :P,R,F:', '%.2f %.2f %.2f'%(all_p*100, all_r*100, all_f*100))
    # print('Results on '+str(span_len)+' F:', '%.2f'%(all_f*100))
    return all_p, all_r, all_f


def find_inwhich_span(span_len, span_ranges, left, right):
    k = (left+right)//2
    if(span_len >= span_ranges[k][0] and span_len <= span_ranges[k][1]):
        return k
    elif(span_len > span_ranges[k][1]):
        return find_inwhich_span(span_len, span_ranges, k+1, right)
    else:
        return find_inwhich_span(span_len, span_ranges, left, k-1)


def top1(lst):
    return max(lst, default='列表为空', key=lambda v: lst.count(v))

def label_mean_width(spans):
    '''
    compute the average width of each label
    '''
    d = defaultdict(list)
    for sent_lst in spans:
        for span in sent_lst:
            label = span[3]
            width = span[2] - span[1] + 1
            d[label].append(width)
    res = {}
    for key, value in dict(d).items():
        avg_width = round(sum(value)/len(value), 1)
        res[key] = avg_width
    print(res)

    major_label = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
    adjust_label = [key for key, value in dict(d).items() if key.startswith('AM')]
    major_lst = []
    adjust_lst = []
    for key, value in dict(d).items():
        if key in major_label:
            major_lst += value
        elif key in adjust_label:
            adjust_lst += value
    major_avg_width = round(sum(major_lst)/len(major_lst), 2)
    adjust_avg_width = round(sum(adjust_lst)/len(adjust_lst), 2)
    print(f'major_avg_width: {major_avg_width} adjust_avg_width: {adjust_avg_width}')
    
    print(f'major label width 众数: {top1(major_lst)}, adjust label width 众数: {top1(adjust_lst)}')
    
    for key, value in dict(d).items():
        t1 = top1(value)
        ratio = value.count(t1)/len(value)
        print(f'{key} 众数: {t1} ratio: {ratio:6.2%}')

    return res

def reorder_label(label_dict):
    itos = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
    for label in label_dict.keys():
        if label not in itos:
            itos.append(label)
    stoi = {}
    for id, value in enumerate(itos):
        stoi[value] = id
    return stoi, itos
    

def cut(spans, label_dict=None, itos=None):
    """
    按照谓词切分
    """
    new_spans = []
    if label_dict == None:
        label_dict = {}
        for sent_span in spans:
            if len(sent_span) == 0:
                new_spans.append([])
                continue
            tmp_dict = {}
            this_res = []
            for span in sent_span:
                p_id = span[0]
                label = span[3]
                if p_id not in tmp_dict.keys():
                    tmp_dict[p_id] = len(tmp_dict)
                    this_res.append([])
                else:
                    this_res[tmp_dict[p_id]].append(span)

                if label not in label_dict.keys():
                    label_dict[label] = len(label_dict)
            new_spans.append(this_res)

        ordered_dict, itos = reorder_label(label_dict)
        return new_spans, ordered_dict, itos
    else:
        for sent_span in spans:
            if len(sent_span) == 0:
                new_spans.append([])
                continue
            tmp_dict = {}
            this_res = []
            for span in sent_span:
                p_id = span[0]
                label = span[3]
                if p_id not in tmp_dict.keys():
                    tmp_dict[p_id] = len(tmp_dict)
                    this_res.append([])
                else:
                    this_res[tmp_dict[p_id]].append(span)

            new_spans.append(this_res)

        return new_spans

def filter_label_cp(p_matrix, itos):
    '''
    delete some label startswith "R" or "C"
    '''
    stack_lst = []
    new_label_lst = []
    new_label_idx = []
    for i, label in enumerate(itos):
        if label[0] != "R" and label[0] != "C":
            new_label_lst.append(label)
            new_label_idx.append(i)
    for idx in new_label_idx:
        stack_lst.append(p_matrix[idx][new_label_idx])
    return torch.stack(stack_lst), new_label_lst
            
def compute_label_cp(spans, label_dict):
    """
    compute label conditional probability in the dataset
    such as p(A1|A2), p(A2|A1)
    """
    num_labels = len(label_dict)
    p_matrix = torch.zeros((num_labels, num_labels), dtype=torch.float)
    for sent_spans in spans:
        if len(sent_spans) == 0:
            continue
        for predicate_spans in sent_spans:
            # 去重，防止：[(10, 7, 7, 'AM-TMP'), (10, 9, 9, 'AM-MNR'), (10, 11, 11, 'A5'), (10, 12, 16, 'A1'), (10, 17, 19, 'AM-TMP')]，导致概率大于1
            span_labels = list(set([span[3] for span in predicate_spans]))

            for label in span_labels:
                label_id = label_dict[label]
                p_matrix[label_id][label_id] += 1
            for i in range(len(span_labels)-1):
                x_label = span_labels[i]
                x_label_id = label_dict[x_label]
                for j in range(i+1, len(span_labels)):
                    y_label = span_labels[j]
                    y_label_id = label_dict[y_label]
                    p_matrix[x_label_id][y_label_id] += 1
                    p_matrix[y_label_id][x_label_id] += 1
    for i in range(num_labels):
        denominator = p_matrix[i][i]
        if denominator == 0:
            p_matrix[i] = 0
        else:
            raw = p_matrix[i].clone()
            p_matrix[i] = raw/denominator
    return p_matrix

                    
def draw_label_heap_map(p_matrix, itos=None):
    p_matrix, label_tag = filter_label_cp(p_matrix, itos)
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.imshow(p_matrix, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    ax.set_xticks(np.arange(p_matrix.shape[1]), minor=False)
    ax.set_yticks(np.arange(p_matrix.shape[0]), minor=False)
    ax.set_xticklabels(label_tag, minor=False, rotation=90, fontsize=7)
    ax.set_yticklabels(label_tag, minor=False, fontsize=7)

    plt.savefig('./heap_map.jpg')

def compute_euc_distance(matrix1, matrix2):
    return (matrix1 - matrix2).pow(2).sum().sqrt().item()

def compute_cos_simi(matrix1, matrix2):
    vec1 = matrix1.reshape(matrix1.shape[0]*matrix1.shape[1])
    vec2 = matrix2.reshape(matrix2.shape[0]*matrix2.shape[1])
    return F.cosine_similarity(vec1, vec2, dim=0).item()

def count_repeat_roles():
    gold_spans = prepare('sc-conll5-dev.conllu')
    new_gold_spans, label_dict, itos = cut(gold_spans)

    sum_p = 0
    repeat_p = 0

    sum_prop = 0
    repeat_prop = 0

    for sent_lst in new_gold_spans:
        for p_lst in sent_lst:
            sum_p += 1
            flag = 0
            exist_role_set = set()
            for prop in p_lst:
                sum_prop += 1
                if prop[3] in exist_role_set:
                    repeat_prop += 1
                    flag = 1
                else:
                    exist_role_set.add(prop[3])
            if flag:
                repeat_p += 1
    print(f'sum_p: {sum_p}, repeat_p: {repeat_p}, ratio: {repeat_p/sum_p:6.2%}')
    print(f'sum_prop: {sum_prop}, repeat_p: {repeat_prop}, ratio: {repeat_prop/sum_prop:6.2%}')

def analy_labels():
    gold_spans = prepare('sc-conll5-train.conllu')
    new_gold_spans, label_dict, itos = cut(gold_spans)
    gold_cp = compute_label_cp(new_gold_spans, label_dict)
    # draw_label_heap_map(label_cp, itos)

    o1_spans = prepare('1o-05-s6-train.pred')
    new_o1_spans = cut(o1_spans, label_dict, itos)
    o1_cp = compute_label_cp(new_o1_spans, label_dict)
    o1_dis = compute_euc_distance(gold_cp, o1_cp)
    print('Euclidean distance:', o1_dis)
    o1_cos_simi = compute_cos_simi(gold_cp, o1_cp)
    print('cosine similarity:', o1_cos_simi)

    o2_spans = prepare('mfvi-05-s6-train.pred')
    new_o2_spans = cut(o2_spans, label_dict, itos)
    o2_cp = compute_label_cp(new_o2_spans, label_dict)
    o2_dis = compute_euc_distance(gold_cp, o2_cp)
    print('Euclidean distance:', o2_dis)
    o2_cos_simi = compute_cos_simi(gold_cp, o2_cp)
    print('cosine similarity:', o2_cos_simi)

    bert_o1_spans = prepare('1o-train-bert-c1.pred')
    new_bert_o1_spans = cut(bert_o1_spans, label_dict, itos)
    bert_o1_cp = compute_label_cp(new_bert_o1_spans, label_dict)
    bert_o1_dis = compute_euc_distance(gold_cp, bert_o1_cp)
    print('Euclidean distance:', bert_o1_dis)
    berto1_cos_simi = compute_cos_simi(gold_cp, bert_o1_cp)
    print('cosine similarity:', berto1_cos_simi)

    bert_o2_spans = prepare('mfvi-train-bert-c1.pred')
    new_bert_o2_spans = cut(bert_o2_spans, label_dict, itos)
    bert_o2_cp = compute_label_cp(new_bert_o2_spans, label_dict)
    bert_o2_dis = compute_euc_distance(gold_cp, bert_o2_cp)
    print('Euclidean distance:', bert_o2_dis)
    berto2_cos_simi = compute_cos_simi(gold_cp, bert_o2_cp)
    print('cosine similarity:', berto2_cos_simi)


def compute_conflict1_inschema3(spans):
    '''
    compute conflict arguments
    an edge has two roles
    '''
    idx = 0
    sum_sents = len(spans)
    conflict_sents = 0
    for sent_spans in spans:
        idx += 1
        all_span_set = set([(span[1], span[2]) for span in sent_spans])
        for span in sent_spans:
            if (span[0], span[1]) in all_span_set:
                conflict_sents += 1
                # print(sent_spans, 'idx:', idx)
                break
    
    print(conflict_sents/sum_sents)
    print(conflict_sents)


def compute_conflict2_inschema3(spans):
    '''
    compute conflict arguments
    different args have same start
    '''
    idx = 0
    sum_args = 0
    conflict_args = 0

    sum_sent = len(spans)
    conflict_sent = 0
    for sent_spans in spans:
        idx += 1
        st_dic = collections.defaultdict(set)
        for span in sent_spans:
            sum_args += 1
            st_dic[span[1]].add(span[2])
        for key, value in st_dic.items():
            if len(value) > 1:
                conflict_sent += 1
                # print(sent_spans, 'idx:', idx, 'key:', key, 'value:', value)
                # break
                # print(sent_spans, 'idx:', idx, 'key:', key, 'value:', value)
                conflict_args += len(value)
            
    
    # print(conflict_sent/sum_sent)
    # print(conflict_sent)

    print(conflict_args/sum_args)
    print(conflict_args)




def del_conflict_inschema3(spans):
    '''
    delete conflict arguments
    '''
    idx = 0
    sum_args = 0
    conflict_args = 0
    res = []
    for sent_spans in spans:
        new_sent_spans = []
        idx += 1
        all_prd_st_set = set([(span[0], span[1]) for span in sent_spans])
        for span in sent_spans:
            sum_args += 1
            if (span[1], span[2]) in all_prd_st_set:
                conflict_args += 1
                print(sent_spans, 'idx:', idx)
            else:
                new_sent_spans.append(span)
        res.append(new_sent_spans)
    
    print(conflict_args/sum_args)
    print(conflict_args)
    return res

def write_schema3_data(filename, n_gold, outfile):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1
    
    with open(outfile, 'w') as f:
        if len(sentences) != len(n_gold):
            print('error')
            return
        for sent, sent_spans in zip(sentences, n_gold):
            line_lsts = [line.split('\t')[0:-2] + ['_', '_'] for line in sent]
            line_edges = [[] for i in range(len(line_lsts))]
            for span in sent_spans:
                p_id, st_id, ed_id, role = span
                p_edge = '0:'+'[prd]'
                if p_edge not in line_edges[p_id-1]:
                    line_edges[p_id-1].append(p_edge)
                
                p_st_edge = str(p_id)+':'+role
                if p_st_edge not in line_edges[st_id-1]:
                    line_edges[st_id-1].append(p_st_edge)
                
                st_ed_edge = str(st_id)+':'+'[spn]'
                if st_ed_edge not in line_edges[ed_id-1]:
                    line_edges[ed_id-1].append(st_ed_edge)
            for i, edge_lst in enumerate(line_edges):
                if len(edge_lst) == 0:
                    continue
                line_lsts[i][-2] = '|'.join(edge_lst)
            
            for line_lst in line_lsts:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')


def write_BIES_data(filename, n_gold, outfile):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1
    
    with open(outfile, 'w') as f:
        if len(sentences) != len(n_gold):
            print('error')
            return
        for sent, sent_spans in zip(sentences, n_gold):
            line_lsts = [line.split('\t')[0:-2] + ['_', '_'] for line in sent]
            line_edges = [[] for i in range(len(line_lsts))]
            for span in sent_spans:
                p_id, st_id, ed_id, role = span
                p_edge = '0:'+'[prd]'
                if p_edge not in line_edges[p_id-1]:
                    line_edges[p_id-1].append(p_edge)
                
                span_length = ed_id-st_id+1
                if span_length == 1:
                    s_str = str(p_id)+':'+'S-'+role
                    line_edges[st_id-1].append(s_str)
                elif span_length == 2:
                    b_str = str(p_id)+':'+'B-'+role
                    e_str = str(p_id)+':'+'E-'+role
                    line_edges[st_id-1].append(b_str)
                    line_edges[ed_id-1].append(e_str)
                else:
                    # span_length>=3
                    b_str = str(p_id)+':'+'B-'+role
                    i_str = str(p_id)+':'+'I-'+role
                    e_str = str(p_id)+':'+'E-'+role
                    line_edges[st_id-1].append(b_str)
                    line_edges[ed_id-1].append(e_str)
                    for tmp_idx in range(st_id, ed_id-1):
                        line_edges[tmp_idx].append(i_str)
            for i, edge_lst in enumerate(line_edges):
                if len(edge_lst) == 0:
                    continue
                line_lsts[i][-2] = '|'.join(edge_lst)
            
            for line_lst in line_lsts:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')

def write_BES_data(filename, n_gold, outfile):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1
    
    with open(outfile, 'w') as f:
        if len(sentences) != len(n_gold):
            print('error')
            return
        for sent, sent_spans in zip(sentences, n_gold):
            line_lsts = [line.split('\t')[0:-2] + ['_', '_'] for line in sent]
            line_edges = [[] for i in range(len(line_lsts))]
            for span in sent_spans:
                p_id, st_id, ed_id, role = span
                p_edge = '0:'+'[prd]'
                if p_edge not in line_edges[p_id-1]:
                    line_edges[p_id-1].append(p_edge)
                
                span_length = ed_id-st_id+1
                if span_length == 1:
                    s_str = str(p_id)+':'+'S-'+role
                    line_edges[st_id-1].append(s_str)
                else:
                    # span_length >= 2
                    b_str = str(p_id)+':'+'B-'+role
                    e_str = str(p_id)+':'+'E-'+role
                    line_edges[st_id-1].append(b_str)
                    line_edges[ed_id-1].append(e_str)
            for i, edge_lst in enumerate(line_edges):
                if len(edge_lst) == 0:
                    continue
                line_lsts[i][-2] = '|'.join(edge_lst)
            
            for line_lst in line_lsts:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')

if __name__ == "__main__":
    gold_spans = prepare('sc-conll5-train.conllu')
    write_BES_data('sc-conll5-train.conllu', gold_spans, 'BES-train.conllu')
    
    
    