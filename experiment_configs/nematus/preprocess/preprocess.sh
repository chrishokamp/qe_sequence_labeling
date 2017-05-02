#!/bin/bash

# this script prepares an APE dataset, including
# preprocessing (tokenization, truecasing, and subword segmentation).
# Note that preprocessing is system and language-pair specific

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

# NOTE: APE 2016 EN-DE data is the same as QE 2016 data, APE 2017 EN-DE data is the same as QE 2017 data
APE_4M_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M
APE_500K_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K
APE_DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

for DIR in $APE_4M_DATADIR $APE_500K_DATADIR $APE_DATADIR
do
    echo "Preparing data in $DIR"
    # NOTE: dev data may not exist in every directory, so errors will print when dev.* isn't found
    for PREFIX in train dev
    do
        # source
        cat ${DIR}/${PREFIX}.src | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
            | ${mosesdecoder}/scripts/recaser/truecase.perl -m $SRC_TRUECASE \
            | ${subword_nmt}/apply_bpe.py -c $SRC_BPE > ${DIR}/${PREFIX}.src.prepped

        # mt
        cat ${DIR}/${PREFIX}.mt | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
            | ${mosesdecoder}/scripts/recaser/truecase.perl -m $TRG_TRUECASE \
            | ${subword_nmt}/apply_bpe.py -c $TRG_BPE > ${DIR}/${PREFIX}.mt.prepped

        # pe
        cat ${DIR}/${PREFIX}.pe | ${mosesdecoder}/scripts/tokenizer/escape-special-chars.perl \
            | ${mosesdecoder}/scripts/recaser/truecase.perl -m $TRG_TRUECASE \
            | ${subword_nmt}/apply_bpe.py -c $TRG_BPE > ${DIR}/${PREFIX}.pe.prepped

    done
    echo "Finished preparing src, trg, mt data in $DIR"

done



