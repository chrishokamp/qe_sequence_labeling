#!/bin/bash

# concat source and mt files, separated by a special token, for use as input into an NMT system
# remember that we need to extract a new source vocabulary for the concatenated input data

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=~/projects/subword_nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=~/projects/nematus

qe_seq_dir=~/projects/qe_sequence_labeling/

# theano device
device=cuda

# NOTE: APE 2016 EN-DE data is the same as QE 2016 data, APE 2017 EN-DE data is the same as QE 2017 data
APE_4M_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
APE_500K_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K/spacy_factor_corpus
APE_DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus

NUM_FACTORS=3

# WORKING: duplicate the join token across all factors
# WORKING: remember that factor corpora also need to be re-joined after mapping segmentation (i.e. join text+factors)
function join_by { local IFS="$1"; shift; echo "$*"; }

concat_token='@BREAK@'
concat_factor_array=()
for ((i=0; i<NUM_FACTORS; i++)); do concat_factor_array+=(${concat_token}); done;

CONCAT_FACTOR=`join_by \|  "${concat_factor_array[@]}"`

#for DIR in $APE_4M_DATADIR $APE_500K_DATADIR $APE_DATADIR
for DIR in $APE_DATADIR
do
    echo "Preparing data in $DIR"
    # NOTE: dev data may not exist in every directory, so errors will print when dev.* isn't found

    for PREFIX in train dev
    do
        # note: /dev/null is used to make paste insert two tabs in a row as the delimiter: https://unix.stackexchange.com/questions/115754/paste-command-setting-multiple-delimiters
        paste ${DIR}/${PREFIX}.en.factors /dev/null ${DIR}/${PREFIX}.de.factors | sed "s/\t\t/ ${CONCAT_FACTOR} /g" > ${DIR}/${PREFIX}.src-mt.concatenated.factors
        echo "Wrote: ${DIR}/${PREFIX}.src-mt.concatenated.factors"
    done
done

# Now extract the nematus vocabulary from the concatenated source-mt
# python $nematus/data/build_dictionary.py ${APE_4M_DATADIR}/train.src-mt.concatenated
# echo "Vocabulary is in ${APE_4M_DATADIR}/train.src-mt.concatenated.json"

