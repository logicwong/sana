#!/usr/bin/env bash

TEXT='../../dataset/WikiBio_sketch'
DEST='../../data-bin/WikiBio_sketch'
python ../../preprocess.py \
-f field -l lpos -r rpos -s value -t skeleton \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir $DEST \
--srcdict $TEXT/dict.sketch.txt --tgtdict $TEXT/dict.sketch.txt \
--workers 64 \
--nwordsfield 6000 --nwordslpos 30 --nwordsrpos 30