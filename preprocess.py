# to transform the raw data to sdp-like data
# 把孤立点还作为孤立点，也就是说待遇和标点符号一样
# 与根的连接，不带词，只有01，02这些

import os
import argparse


def read_data(file_name):
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    # for s in sentences:
    #     if(len(s) < 4):
    #         print(s)
    return sentences


def write_data(suffix_name, sentences):
    path_name, base_name = os.path.split(suffix_name)
    new_file_name = 'new_' + base_name
    f = open(new_file_name, 'w')
    for sentence in sentences:
        # [line1,line2,...]
        predicate_map = dict()
        # sense_map = dict()
        new_sentence = []
        for i, line in enumerate(sentence, 1):
            line_lst = line.split('\t')
            new_line_lst = [
                line_lst[0], line_lst[1], line_lst[2], "_", line_lst[4], "_",
                "_", "_", "_", "_"
            ]
            if (line_lst[12] != "_"):
                predicate_map[len(predicate_map) + 1] = i
                # sense_map[len(sense_map)+1] = line_lst[13]
            new_sentence.append(new_line_lst)

        for i, line in enumerate(sentence, 1):
            line_lst = line.split('\t')
            arc_values = []
            if (line_lst[12] != "_"):
                arc_values.append('0:' + line_lst[13].split('.')[1])
            for j in range(14, len(line_lst)):
                if (line_lst[j] != '_'):
                    head_idx = predicate_map[j - 14 + 1]
                    arc_values.append(str(head_idx) + ':' + line_lst[j])
            if (len(arc_values) > 0):
                value = '|'.join(arc_values)
                new_sentence[i - 1][-2] = value

        for new_s_lst in new_sentence:
            f.write('\t'.join(new_s_lst) + '\n')
        f.write('\n')

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='to transform the raw data to sdp-like data.')
    parser.add_argument('--file', help='file name.')
    args = parser.parse_args()

    # pdb.set_trace()

    file_name = args.file
    sentences = read_data(file_name)
    write_data(file_name, sentences)