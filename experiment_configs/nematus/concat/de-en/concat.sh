#!/bin/bash

# concat source and mt files, separated by a special token, for use as input into an NMT system
# WORKING HERE
# TODO: extract a new source vocabulary for the concatenated input data
# TODO: the delimiter between src-mt is particularly important

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
APE_4M_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M
APE_500K_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K
APE_DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

for DIR in $APE_4M_DATADIR $APE_500K_DATADIR $APE_DATADIR
do
    echo "Preparing data in $DIR"
    # NOTE: dev data may not exist in every directory, so errors will print when dev.* isn't found

    for PREFIX in train dev
    do
        # note: /dev/null is used to make paste insert two tabs in a row as the delimiter: https://unix.stackexchange.com/questions/115754/paste-command-setting-multiple-delimiters
        paste ${DIR}/${PREFIX}.src.prepped /dev/null ${DIR}/${PREFIX}.mt.prepped | sed 's/\t\t/ \@BREAK\@ /g' > ${DIR}/${PREFIX}.src-mt.concatenated
        echo "Wrote: ${DIR}/${PREFIX}.src-mt.concatenated"
    done
done

# Now extract the nematus vocabulary from the concatenated source-mt
python $nematus/data/build_dictionary.py ${APE_4M_DATADIR}/train.src-mt.concatenated
echo "Vocabulary is in ${APE_4M_DATADIR}/train.src-mt.concatenated.json"


