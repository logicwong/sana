TEXT='../../dataset/WikiBio_fairseq'
DEST='../../dataset/WikiBio_sketch'
mkdir $DEST

vocab_size=25000
position_markers=1000
export LC_ALL=C
cat $TEXT/train.tgt |
tr -s '[:space:]' '\n' |
sort |
uniq -c |
sort -k1,1bnr -k2 |
head -n "$((vocab_size - 4))" |
awk '{ print $2 " " $1 }' > $DEST/dict.sketch.txt
python3 -c "[print('<{}> 0'.format(word)) for word in ['ins', 'sep']]" >> $DEST/dict.sketch.txt
python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >> $DEST/dict.sketch.txt

python3 ../../preprocess_ptr.py \
--source $TEXT/train.value --target $TEXT/train.skeleton \
--vocab $DEST/dict.sketch.txt \
--source-out $DEST/train.value --target-out $DEST/train.skeleton \

python3 ../../preprocess_ptr.py \
--source $TEXT/valid.value --target $TEXT/valid.skeleton \
--vocab $DEST/dict.sketch.txt \
--source-out $DEST/valid.value --target-out $DEST/valid.skeleton \

python3 ../../preprocess_ptr.py \
--source $TEXT/test.value --target $TEXT/test.skeleton \
--vocab $DEST/dict.sketch.txt \
--source-out $DEST/test.value --target-out $DEST/test.skeleton \

rsync -avr --exclude='train.value' --exclude='train.skeleton' --exclude='train.tgt' \
--exclude='valid.value' --exclude='valid.skeleton' --exclude='valid.tgt' \
--exclude='test.value' --exclude='test.skeleton' --exclude='test.tgt' \
$TEXT/train.* $TEXT/valid.* $TEXT/test.* $DEST