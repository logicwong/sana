#!/usr/bin/env bash

DATA='../../data-bin/WikiBio_sketch'
SAVE='../../checkpoints/WikiBio_sketch'
USER='../../AR_data2text/sana_module'
RESULT='../../results/WikiBio_sketch'
OUTPUT='../../dataset/WikiBio_fairseq'

CUDA_VISIBLE_DEVICES=0 python3 ../../generate.py $DATA \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --target-lang skeleton \
--user-dir $USER \
--task data2text \
--path $SAVE/checkpoint_best.pt \
--scoring sacrebleu \
--beam 5 \
--no-repeat-ngram-size 3 \
--max-len-b 40 \
--max-len-a 0 \
--max-tokens 8000 \
--gen-subset train \
--results-path $RESULT

CUDA_VISIBLE_DEVICES=0 python3 ../../generate.py $DATA \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --target-lang skeleton \
--user-dir $USER \
--task data2text \
--path $SAVE/checkpoint_best.pt \
--scoring sacrebleu \
--beam 5 \
--no-repeat-ngram-size 3 \
--max-len-b 40 \
--max-len-a 0 \
--max-tokens 8000 \
--gen-subset valid \
--results-path $RESULT

CUDA_VISIBLE_DEVICES=0 python3 ../../generate.py $DATA \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --target-lang skeleton \
--user-dir $USER \
--task data2text \
--path $SAVE/checkpoint_best.pt \
--scoring sacrebleu \
--beam 5 \
--lenpen 1.1 \
--no-repeat-ngram-size 3 \
--max-len-b 40 \
--max-len-a 0 \
--max-tokens 8000 \
--gen-subset test \
--results-path $RESULT

python ../../dataset/extract_gen_skeleton.py $RESULT $OUTPUT 'train'
python ../../dataset/extract_gen_skeleton.py $RESULT $OUTPUT 'valid'
python ../../dataset/extract_gen_skeleton.py $RESULT $OUTPUT 'test'
