#!/usr/bin/env bash

# the files we look for are ${PREFIX}.src.prepped and ${PREFIX}.mt.prepped
PREFIX=$1
# Input is directory which contains files named test.src and test.mt
DATADIR=$2

printf "Looking for set: $PREFIX in: $DATADIR\n"
source activate theano

# suffix of source language
SRC=en

# suffix of target language
TRG=de

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=~/projects/subword_nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=~/projects/nematus

qe_seq_dir=~/projects/qe_sequence_labeling/

# theano device
device=cuda

# This is the model we use to obtain MT-SRC alignments
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
MODEL_PREFIX=${MODEL_DIR}/model.iter

# we need to go to the dir because the paths in the <config>.json files are local to ${MODEL_DIR}
cd ${MODEL_DIR}

# Note: 4 models are available, but we just use the last one, because we only need one set of alignments
# ${MODEL_PREFIX}{340000,350000,360000,370000}.npz \

echo "Preparing data in $DATADIR"
# NOTE: we're actually getting alignments only from the first model in a 4-model ensemble
FAKE_NBEST=$DATADIR/$PREFIX.mt.fake_nbest
# add numbers
awk '{ print FNR - 1 " ||| " $0 }' $DATADIR/$PREFIX.mt.prepped > $FAKE_NBEST

# extract alignments
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/rescore.py \
 -b 16 \
 -n \
 -s ${DATADIR}/${PREFIX}.src.prepped \
 -i $FAKE_NBEST \
 -o ${DATADIR}/${PREFIX}.mt.rescore_output \
 -m ${MODEL_PREFIX}370000.npz \
 --walign

echo "Finished computing alignments between $DATADIR/$PREFIX.src.prepped and $DATADIR/$PREFIX.mt.prepped"

# NOW EXTRACT ALIGNMENT CORPUS
python $qe_seq_dir/scripts/alignment_corpus_from_nematus_json_output.py --json ${DATADIR}/${PREFIX}.mt.rescore_output_withwords.json --output ${DATADIR}/${PREFIX}.mt_source_aligned --order target
# Note: some rows have parsing errors, we need to keep track of these!!
# read the deleted rows file, also delete these rows from (src, mt, pe prepped data)
readarray -t DELETE_LINES < ${DATADIR}/${PREFIX}.mt.rescore_output_withwords.json.deleted_rows
echo 'I want to delete the following lines from *.src.*, *.mt.* and *.pe.*: '
echo ${DELETE_LINES[@]}

# This is to remove broken lines, but we don't want to do this for dev and test sets
#cp ${DATADIR}/${PREFIX}.src.prepped ${DATADIR}/${PREFIX}.src.prepped.orig
#cp ${DATADIR}/${PREFIX}.mt.prepped ${DATADIR}/${PREFIX}.mt.prepped.orig
#cp ${DATADIR}/${PREFIX}.pe.prepped ${DATADIR}/${PREFIX}.pe.prepped.orig
#
#COUNTER=0
#for LINE_NO in ${DELETE_LINES[@]}
#do
#    ACTUAL_LINE_NO=`expr $LINE_NO - $COUNTER`
#    echo "Deleting line: $ACTUAL_LINE_NO"
#    sed  -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.src.prepped
#    sed -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.mt.prepped
#    sed -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.pe.prepped
#    COUNTER=$((COUNTER + 1))
#done

python $qe_seq_dir/scripts/create_factor_corpus.py --f1 ${DATADIR}/${PREFIX}.mt.prepped --f2 ${DATADIR}/${PREFIX}.mt_source_aligned --output ${DATADIR}/${PREFIX}.mt.factor_corpus



