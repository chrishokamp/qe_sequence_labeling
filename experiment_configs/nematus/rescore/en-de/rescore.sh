#!/bin/bash

# this sample script prepares a dataset, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# Then performs forced decoding, and outputs a json file with alignments

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

# WORKING: follow rescore.sh in scoring and extracting alignments for {train,dev} simultaeneously

# NOTE: APE 2016 EN-DE data is the same as QE 2016 data, APE 2017 EN-DE data is the same as QE 2017 data
APE_4M_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M
APE500K_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K
APE_DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

# This is the model we use to obtain MT-SRC alignments
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
MODEL_PREFIX=${MODEL_DIR}/model.iter

# we need to go to the dir because the paths in the <config>.json files are local to ${MODEL_DIR}
cd ${MODEL_DIR}

for DIR in $APE_4M_DATADIR $APE500k_DATADIR $APE_DATADIR
do
    echo "Preparing data in $DIR"
    # NOTE: dev data may not exist in every directory, so errors will print when dev.* isn't found
    # NOTE: even though we're loading every model, we're actually getting alignments only from the first one

    for PREFIX in train dev
    do
        FAKE_NBEST=$DIR/$PREFIX.mt.fake_nbest
        # add numbers
        awk '{ print FNR - 1 " ||| " $0 }' $DIR/$PREFIX.mt.prepped > $FAKE_NBEST
        # extract alignments
        THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/rescore.py \
         -b 64 \
         -n \
         -s ${DIR}/${PREFIX}.src.prepped \
         -i $FAKE_NBEST \
         -o ${DIR}/${PREFIX}.mt.rescore_output \
         -m ${MODEL_PREFIX}{340000,350000,360000,370000}.npz \
         --walign

         echo "Finished computing alignments between $DIR/$PREFIX.src.prepped and $DIR/$PREFIX.mt.prepped"

         # NOW EXTRACT ALIGNMENT CORPUS
         python $qe_seq_dir/scripts/alignment_corpus_from_nematus_json_output.py --json ${DIR}/${PREFIX}.mt.rescore_output_withwords.json --output ${DIR}/${PREFIX}.mt_source_aligned --order target
         python $qe_seq_dir/scripts/create_factor_corpus.py --f1 ${DIR}/${PREFIX}.mt.prepped --f2 ${DIR}/${PREFIX}.mt_source_aligned --output ${DIR}/${PREFIX}.mt.factor_corpus

    done

done



echo "Finished preparing fake MT one-best list"



