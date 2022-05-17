#!/usr/bin/env bash

DATA='../../data-bin/WikiBio_refinement'
SAVE='../../checkpoints/WikiBio_refinement'
USER='../../sana_module'
RESULT='../../results/WikiBio_refinement'
CUDA_VISIBLE_DEVICES=0 python ../../generate.py $DATA \
--field-lang field --lpos-lang lpos --rpos-lang rpos --source-lang value --kwd-lang gen_skeleton --target-lang tgt \
--user-dir $USER \
--task refinement \
--path $SAVE/checkpoint_best.pt \
--scoring sacrebleu \
--seed 7 \
--eval-bleu-args '{"no_delete_kwd": true}' \
--max-tokens 8000 \
--gen-subset test \
--results-path $RESULT
python ../../dataset/extract_gen_tgt.py $RESULT $RESULT 'test'

VALUE='../../dataset/WikiBio_fairseq/test.value'
REFERENCES_PATH='../../dataset/WikiBio_fairseq/test.tgt'
PREDICTION_PATH=$RESULT/wb_predictions.txt
TABLES_PATH=../../utils/parent/wb_test_tables.txt
python ../../postprocess_ptr.py --source $VALUE --target $RESULT/test.gen_tgt --target-out $PREDICTION_PATH
python ../../utils/parent/table_text_eval.py --generations $PREDICTION_PATH --references $REFERENCES_PATH --tables $TABLES_PATH
python ../../utils/parent/table_text_eval.py --generations $PREDICTION_PATH --references $REFERENCES_PATH --tables $TABLES_PATH --PARENT_T true