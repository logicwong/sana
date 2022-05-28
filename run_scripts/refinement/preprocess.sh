#!/usr/bin/env bash

TEXT='../../dataset/WikiBio_refinement'
DEST='../../data-bin/WikiBio_refinement'
python ../../preprocess.py \
-f field -l lpos -r rpos -s value -k gen_skeleton -t tgt \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir $DEST \
--srcdict $TEXT/dict.refine.txt --tgtdict $TEXT/dict.refine.txt \
--workers 64 \
--nwordsfield 6000
