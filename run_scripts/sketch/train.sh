#!/usr/bin/env bash

DATA='../../data-bin/WikiBio_sketch'
SAVE='../../checkpoints/WikiBio_sketch'
USER='../../AR_data2text/sana_module'

CUDA_VISIBLE_DEVICES=0,1 python3 ../../train.py $DATA \
--save-dir $SAVE --user-dir $USER \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --target-lang skeleton \
--task data2text --arch sketch_base --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.3 --weight-decay 0.01 \
--ddp-backend=no_c10d \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--decoder-learned-pos --encoder-learned-pos \
--log-interval 100 --log-format 'simple' \
--max-tokens 16384 \
--update-freq 2 --validate-interval 1 --validate-after-updates 4000 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 0, "max_len_b": 40, "no_repeat_ngram_size": 3}' \
--eval-tokenized-bleu \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--patience 15 \
--keep-last-epochs 10 \
--source-position-markers 1000 \
--fixed-validation-seed=7 \
--eval-bleu-print-samples
