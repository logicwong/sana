#!/usr/bin/env bash

DATA='../../data-bin/WikiBio_refinement'
SAVE='../../checkpoints/WikiBio_refinement'
USER='../../sana_module'

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../train.py $DATA \
--save-dir $SAVE --user-dir $USER \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --kwd-lang gen_skeleton --target-lang tgt \
--ddp-backend=no_c10d --task refinement \
--criterion adjust_nat_loss --arch refinement_base --noise random_delete \
--share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
--warmup-updates 10000 --warmup-init-lr '1e-07' --label-smoothing 0.1 \
--dropout 0.3 --weight-decay 0.01 \
--decoder-learned-pos --encoder-learned-pos \
--apply-bert-init --log-format 'simple' \
--log-interval 100 --fixed-validation-seed 7 \
--max-tokens 16382 --max-update 300000 \
--update-freq 1 --save-interval 50 --validate-interval 50 \
--save-interval-updates 10000 --validate-interval-updates 10000 \
--eval-bleu \
--eval-bleu-args '{"no_delete_kwd": true}' \
--eval-tokenized-bleu \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--patience 100 --beta 0.7 \
--keep-last-epochs 15 \
--fp16 \
--source-position-markers 1000 \
--eval-bleu-print-samples