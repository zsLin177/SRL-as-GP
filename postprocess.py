# change conllu to raw file type
import argparse
import pdb


def poprocess(gold_file, pred_file):
    '''
    gold_file : just for padding synatic content
    pred_file : type of conllu
    '''
    false_num = 0

    with open(gold_file, 'r') as f:
        lines = [line.strip() for line in f]
        gold_sentences = []
        start, i = 0, 0
        for line in lines:
            if not line:
                # sentences.append(lines[start:i])
                gold_sentences.append(
                    [l_str.split('\t') for l_str in lines[start:i]])
                start = i + 1
            i += 1
        # sentences:[[[],[]],[]]

    with open(pred_file, 'r') as f:
        lines = [line.strip() for line in f]
        pred_sentences = []
        start, i = 0, 0
        for line in lines:
            if not line:
                # sentences.append(lines[start:i])
                pred_sentences.append(
                    [l_str.split('\t') for l_str in lines[start:i]])
                start = i + 1
            i += 1

    if (len(gold_sentences) == len(pred_sentences)):
        template = ['_', '_']
        new_sentences = []
        for i in range(len(gold_sentences)):
            gold_s = gold_sentences[i]
            pred_s = pred_sentences[i]
            new_s = []
            for word in gold_s:
                new_s.append(word[0:12] + template)

            arc_values = []
            for pred_word in pred_s:
                arc_values.append(pred_word[8])

            predicate_map = {}
            arc_lsts = []
            for arc_value in arc_values:
                if (arc_value == '_'):
                    arc_lsts.append([])
                else:
                    arc_lsts.append(
                        [pair.split(':') for pair in arc_value.split('|')])

            arc_vocabs_lst = []
            for k, arc_lst in enumerate(arc_lsts, 1):
                if (len(arc_lst) == 0):
                    arc_vocab = {}
                    arc_vocabs_lst.append(arc_vocab)
                else:
                    arc_vocab = {}
                    for arc in arc_lst:
                        if (arc[0] == '0'):
                            predicate_map[k] = len(predicate_map)
                            new_s[k - 1][12] = 'Y'
                            new_s[k - 1][13] = new_s[k - 1][2] + '.' + arc[1]
                        else:
                            arc_vocab[arc[0]] = arc[1]

                    arc_vocabs_lst.append(arc_vocab)

            arc_template = ['_'] * len(predicate_map)
            for k, arc_vocab in enumerate(arc_vocabs_lst, 1):
                new_s[k - 1] += arc_template
                if (len(arc_vocab) > 0):
                    for key, value in arc_vocab.items():
                        # 如果预测出来的头没有被识别成谓词，那这个边就不要
                        if (int(key) in list(predicate_map.keys())):
                            bias = predicate_map[int(key)]
                            new_s[k - 1][14 + bias] = value
                        else:
                            false_num += 1

            new_sentences.append(new_s)

        with open(pred_file + '.conll09', 'w') as f:
            for new_s in new_sentences:
                for line in new_s:
                    new_line = '\t'.join(line)
                    f.write(new_line + '\n')
                f.write('\n')

        print(false_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='to transform the raw data to raw data.')
    parser.add_argument('--gold', help='file name.')
    parser.add_argument('--pred', help='file name.')
    args = parser.parse_args()

    # pdb.set_trace()

    gold_name = args.gold
    pred_file = args.pred
    poprocess(gold_name, pred_file)