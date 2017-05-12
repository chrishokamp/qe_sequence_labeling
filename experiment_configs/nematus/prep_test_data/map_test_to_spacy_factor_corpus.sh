#!/usr/bin/env bash

# the files we look for are ${PREFIX}.src.prepped and ${PREFIX}.mt.prepped
PREFIX=$1
# Input is directory which contains files named test.src and test.mt
DATADIR=$2

printf "Looking for set: $PREFIX in: $DATADIR\n"

source activate spacy
QE_SEQ=~/projects/qe_sequence_labeling

mkdir -p $DATADIR/spacy_factor_corpus

# extract factors
# src
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/${PREFIX}.src -o $DATADIR/spacy_factor_corpus -l en -p ${PREFIX}
# mt
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/${PREFIX}.mt -o $DATADIR/spacy_factor_corpus -l de -p ${PREFIX}

# rename datadir
DATADIR=$DATADIR/spacy_factor_corpus

VOCAB_DIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus

# apply subword segmentation to text factor
SUBWORD=~/projects/subword_nmt
NUM_OPERATIONS=40000
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.en --vocabulary-threshold 50 < $DATADIR/${PREFIX}.en.tok > $DATADIR/${PREFIX}.en.bpe
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.de --vocabulary-threshold 50 < $DATADIR/${PREFIX}.de.tok > $DATADIR/${PREFIX}.de.bpe

SEQUENCE_QE=~/projects/qe_sequence_labeling

# segment factors by text factor segmentation
# src
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/${PREFIX}.en.bpe -f $DATADIR/${PREFIX}.en.factors
# mt
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/${PREFIX}.de.bpe -f $DATADIR/${PREFIX}.de.factors

# Concat text factors
# note: /dev/null is used to make paste insert two tabs in a row as the delimiter: https://unix.stackexchange.com/questions/115754/paste-command-setting-multiple-delimiters
paste ${DATADIR}/${PREFIX}.en.bpe /dev/null ${DATADIR}/${PREFIX}.de.bpe | sed 's/\t\t/ \@BREAK\@ /g'  > ${DATADIR}/${PREFIX}.src-mt.concatenated.bpe
echo "Wrote: ${DATADIR}/${PREFIX}.src-mt.concatenated.bpe"

# Concat Spacy factors
NUM_FACTORS=3

function join_by { local IFS="$1"; shift; echo "$*"; }

concat_token='@BREAK@'
concat_factor_array=()
for ((i=0; i<NUM_FACTORS; i++)); do concat_factor_array+=(${concat_token}); done;

CONCAT_FACTOR=`join_by \|  "${concat_factor_array[@]}"`

# note: /dev/null is used to make paste insert two tabs in a row as the delimiter: https://unix.stackexchange.com/questions/115754/paste-command-setting-multiple-delimiters
paste ${DATADIR}/${PREFIX}.en.factors.bpe /dev/null ${DATADIR}/${PREFIX}.de.factors.bpe | sed "s/\t\t/ ${CONCAT_FACTOR} /g" > ${DATADIR}/${PREFIX}.src-mt.concatenated.factors.bpe
echo "Wrote: ${DATADIR}/${PREFIX}.src-mt.concatenated.factors.bpe"

# join text factor with Spacy factor
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/${PREFIX}.src-mt.concatenated.bpe -f $DATADIR/${PREFIX}.src-mt.concatenated.factors.bpe

printf "Final mapped factor corpus is in: ${DATADIR}/${PREFIX}.src-mt.factor_corpus\n"
