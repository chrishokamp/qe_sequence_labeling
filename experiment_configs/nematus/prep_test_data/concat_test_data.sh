#!/usr/bin/env bash

# concat source and mt files, separated by a special token, for use as input into an NMT system
# remember that we need to extract a new source vocabulary for the concatenated input data

# the files we look for are ${PREFIX}.src.prepped and ${PREFIX}.mt.prepped
PREFIX=$1
# Input is directory which contains files named test.src and test.mt
DATADIR=$2

printf "Looking for set: $PREFIX in: $DATADIR\n"

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=~/projects/subword_nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=~/projects/nematus

qe_seq_dir=~/projects/qe_sequence_labeling/

# theano device
device=cuda

printf "Preparing data in $DATADIR\n"
# note: /dev/null is used to make paste insert two tabs in a row as the delimiter: https://unix.stackexchange.com/questions/115754/paste-command-setting-multiple-delimiters
paste ${DATADIR}/${PREFIX}.src.prepped /dev/null ${DATADIR}/${PREFIX}.mt.prepped | sed 's/\t\t/ \@BREAK\@ /g' > ${DATADIR}/${PREFIX}.src-mt.concatenated
printf "Wrote: ${DATADIR}/${PREFIX}.src-mt.concatenated\n"

