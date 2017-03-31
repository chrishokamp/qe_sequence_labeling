#!/bin/bash

# this sample script translates a test set, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# and postprocessing (merging subword units, detruecasing, detokenization).

# instructions: set paths to mosesdecoder, subword_nmt, and nematus,
# then run "./translate.sh < input_file > output_file"

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

#DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16
DATADIR=/media/1tb_drive/Dropbox/data/qe/wmt_2016/dev_wmt16_pretrained_bpe
ORIG_SRC_FILE=$DATADIR/dev.src
PREPPED_SRC_FILE=$DATADIR/dev.src.prepped

ORIG_TRG_FILE=$DATADIR/dev.mt
PREPPED_TRG_FILE=$DATADIR/dev.mt.prepped
TRG_FILE=$DATADIR/dev.mt.numbered
OUTPUT_FILE=$DATADIR/dev.mt.rescored

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG < $ORIG_TRG_FILE > ${ORIG_TRG_FILE}_1
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $TRG -penn < ${ORIG_TRG_FILE}_1 > ${ORIG_TRG_FILE}_2
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$TRG < ${ORIG_TRG_FILE}_2 > ${ORIG_TRG_FILE}_3
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < ${ORIG_TRG_FILE}_3 > $PREPPED_TRG_FILE

echo "Finished prepping MT data"

# add numbers to MT to simulate n-best list
awk '{ print FNR - 1 " ||| " $0 }' $PREPPED_TRG_FILE > $TRG_FILE

echo "Finished preparing fake MT one-best list"

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC < $ORIG_SRC_FILE > ${ORIG_SRC_FILE}_1
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $SRC -penn < ${ORIG_SRC_FILE}_1 > ${ORIG_SRC_FILE}_2
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC < ${ORIG_SRC_FILE}_2 > ${ORIG_SRC_FILE}_3
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < ${ORIG_SRC_FILE}_3 > $PREPPED_SRC_FILE

# theano device
device=cuda

# preprocess
# rescore and output alignments
#     -m model-ens{1,2,3,4}.npz \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/rescore.py \
     -b 1 \
     -n \
     -s $PREPPED_SRC_FILE \
     -i $TRG_FILE \
     -o $OUTPUT_FILE \
     -m model-ens1.npz \
     --walign
# postprocess
#sed 's/\@\@ //g' | \
#$mosesdecoder/scripts/recaser/detruecase.perl | \
#$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG

