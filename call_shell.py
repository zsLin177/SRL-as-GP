import subprocess
import pdb
# import warnings
# warnings.filterwarnings("ignore")

_SRL_CONLL_EVAL_SCRIPT = 'conll05-original-style/eval.sh'

def get_results(gold_path, pred_path):
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, pred_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]

    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, pred_path, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    # pdb.set_trace()
    # temp = str(eval_info).strip().split("\\n")
    conll_recall = float(str(eval_info).strip().split("\\n")[-42:][6].strip().split()[5])
    conll_precision = float(str(eval_info2).strip().split("\\n")[-42:][6].strip().split()[5])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("\\n")[-42:][6].strip().split()[6])

    return conll_f1, lisa_f1

if __name__ == '__main__':
    gold_path = 'wsj-gold.final'
    pred_path = 'wsj-gold.final'
    print(get_results(gold_path, pred_path))


