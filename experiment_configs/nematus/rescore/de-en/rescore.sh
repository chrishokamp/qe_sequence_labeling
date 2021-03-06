#!/bin/bash

# this sample script translates a test set, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# and postprocessing (merging subword units, detruecasing, detokenization).

# instructions: set paths to mosesdecoder, subword_nmt, and nematus,
# then run "./translate.sh < input_file > output_file"

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=~/projects/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=~/projects/nematus

# suffix of source language
SRC=de

# suffix of target language
TRG=en

DATADIR=/extra/chokamp/qe_data/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16
#ORIG_SRC_FILE=$DATADIR/train.src.small
ORIG_SRC_FILE=$DATADIR/train.mt
PREPPED_SRC_FILE=$DATADIR/train.rescore.preprocessed.mt
#ORIG_TRG_FILE=$DATADIR/train.mt.small
ORIG_TRG_FILE=$DATADIR/train.src
PREPPED_TRG_FILE=$DATADIR/train.rescore.preprocessed.src
TRG_FILE=$DATADIR/train.src.numbered
OUTPUT_FILE=$DATADIR/train.src.rescored

#$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$TRG | \

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG < $ORIG_TRG_FILE > ${ORIG_TRG_FILE}_1
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $TRG -penn < ${ORIG_TRG_FILE}_1 > ${ORIG_TRG_FILE}_2
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < ${ORIG_TRG_FILE}_2 > $PREPPED_TRG_FILE

echo "Finished prepping MT data"

# add numbers
awk '{ print FNR - 1 " ||| " $0 }' $PREPPED_TRG_FILE > $TRG_FILE

echo "Finished preparing fake MT one-best list"

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC < $ORIG_SRC_FILE > ${ORIG_SRC_FILE}_1
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $SRC -penn < ${ORIG_SRC_FILE}_1 > ${ORIG_SRC_FILE}_2
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC < ${ORIG_SRC_FILE}_2 > ${ORIG_SRC_FILE}_3
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < ${ORIG_SRC_FILE}_3 > $PREPPED_SRC_FILE

# theano device
device=gpu

# preprocess
# rescore and output alignments
#     -m model-ens{1,2,3,4}.npz \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/rescore.py \
     -b 1 \
     -n \
     -s $PREPPED_SRC_FILE \
     -i $TRG_FILE \
     -o $OUTPUT_FILE \
     -m model.npz \
     --walign
# postprocess
#sed 's/\@\@ //g' | \
#$mosesdecoder/scripts/recaser/detruecase.perl | \
#$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG

