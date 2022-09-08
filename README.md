# Fast and Accurate End-to-End Span-based Semantic Role Labeling as Word-based Graph Parsing
This is the repo for SRLasSDGP, a novel approach to form span-based SRL as word-based graph parsing, to be presented at [COLING2022](https://coling2022.org/coling). A preprint of the paper can be found at the [following location on arxiv](https://arxiv.org/abs/2112.02970).

## Installation
```shell
pip install -r  requirements.txt
```

## Preprocess data
Under construction.

## Training and Prediction
### Training models without pre-trained language models
```
python -m supar.cmds.vi_SRL train -b \
        --train data/BES/conll05/BES-train.conllu \
        --dev data/BES/conll05/BES-dev.conllu \
        --test data/BES/conll05/BES-wsj.conllu \
        --batch-size 3000 \
        --n-embed 100 \
        --feat lemma,char \
        --itp 0.06 \
        --embed data/glove.6B.300d.txt \
        --n_pretrained_embed 300 \
        --min_freq 7 \
        --split \
        --seed 1 \
        --schema BES \
        -p exp/exp1/model \
        -d 5
```
* You can add the "--train_given_prd" in the script to train a predicate-given model.
* "--schema" to specify which schema to use.
* "-d" to specify which gpu to use.
* Detailed description about args can be found in "supar/cmds/cmd.py" and "supar/cmds/vi_SRL.py".

### Prediction and Evaluation
```
python -m supar.cmds.vi_SRL predict --data data/BES/conll05/BES-wsj.conllu \
        -p exp/exp1/model \
        --pred BES-wsj.pred \
        --gold data/conll05-original-style/wsj.final \
        --task 05 \
        --schema BES \
        --vtb \
        -d 4
```
* "--vtb" means whether to use constrained viterbi.

### Training models with pre-trained language models



