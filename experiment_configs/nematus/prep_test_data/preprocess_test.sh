#!/bin/bash

# this script prepares an APE dataset, including
# preprocessing (tokenization, truecasing, and subword segmentation).
# Note that preprocessing is system and language-pair specific

# the files we look for are ${PREFIX}.src and ${PREFIX}.mt
PREFIX=$1
# Input is directory which contains files named test.src and test.mt
DATADIR=$2

printf "Looking for set: $PREFIX in: $DATADIR\n"

# suffix of source language
SRC=en

# suffix of target language
TRG=de

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=~/projects/subword_nmt

# path to nematus (https://www.github.com/rsennrich/nematus)
nematus=~/projects/nematus

SRC_BPE=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/data/en.bpe
TRG_BPE=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/data/de.bpe
SRC_TRUECASE=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/data/true.en
TRG_TRUECASE=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/data/true.de

echo "Preparing data in $DATADIR"
# source
cat ${DATADIR}/${PREFIX}.src | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
    | ${mosesdecoder}/scripts/recaser/truecase.perl -m $SRC_TRUECASE \
    | ${subword_nmt}/apply_bpe.py -c $SRC_BPE > ${DATADIR}/${PREFIX}.src.prepped

# mt
cat ${DATADIR}/${PREFIX}.mt | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
    | ${mosesdecoder}/scripts/recaser/truecase.perl -m $TRG_TRUECASE \
    | ${subword_nmt}/apply_bpe.py -c $TRG_BPE > ${DATADIR}/${PREFIX}.mt.prepped

## pe
#cat ${DATADIR}/${PREFIX}.pe | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
#    | ${mosesdecoder}/scripts/recaser/truecase.perl -m $TRG_TRUECASE \
#    | ${subword_nmt}/apply_bpe.py -c $TRG_BPE > ${DATADIR}/${PREFIX}.pe.prepped

printf "Finished preparing data\n"



