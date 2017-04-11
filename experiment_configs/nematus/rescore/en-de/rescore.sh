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

# NOTE: APE 2016 EN-DE data is the same as QE 2016 data, APE 2017 EN-DE data is the same as QE 2017 data
APE_4M_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M
APE_500K_DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K
APE_DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

# This is the model we use to obtain MT-SRC alignments
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
MODEL_PREFIX=${MODEL_DIR}/model.iter

# we need to go to the dir because the paths in the <config>.json files are local to ${MODEL_DIR}
cd ${MODEL_DIR}

# Note: 4 models are available, but we just use the last one, because we only need one set of alignments
# ${MODEL_PREFIX}{340000,350000,360000,370000}.npz \

for DIR in $APE_4M_DATADIR $APE_500K_DATADIR $APE_DATADIR
do
    echo "Preparing data in $DIR"
    # NOTE: dev data may not exist in every directory, so errors will print when dev.* isn't found
    # NOTE: we're actually getting alignments only from the first model in a 4-model ensemble

    for PREFIX in train dev
    do
        FAKE_NBEST=$DIR/$PREFIX.mt.fake_nbest
        # add numbers
        awk '{ print FNR - 1 " ||| " $0 }' $DIR/$PREFIX.mt.prepped > $FAKE_NBEST

        # extract alignments
        THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/rescore.py \
         -b 16 \
         -n \
         -s ${DIR}/${PREFIX}.src.prepped \
         -i $FAKE_NBEST \
         -o ${DIR}/${PREFIX}.mt.rescore_output \
         -m ${MODEL_PREFIX}370000.npz \
         --walign

        echo "Finished computing alignments between $DIR/$PREFIX.src.prepped and $DIR/$PREFIX.mt.prepped"

        # NOW EXTRACT ALIGNMENT CORPUS
        python $qe_seq_dir/scripts/alignment_corpus_from_nematus_json_output.py --json ${DIR}/${PREFIX}.mt.rescore_output_withwords.json --output ${DIR}/${PREFIX}.mt_source_aligned --order target
        # Note: some rows have parsing errors, we need to keep track of these!!
        # read the deleted rows file, also delete these rows from (src, mt, pe prepped data)
        readarray -t DELETE_LINES < ${DIR}/${PREFIX}.mt.rescore_output_withwords.json.deleted_rows
        echo 'I will delete the following lines from *.src.*, *.mt.* and *.pe.*: '
        echo ${DELETE_LINES[@]}

        cp ${DIR}/${PREFIX}.src.prepped ${DIR}/${PREFIX}.src.prepped.orig 
        cp ${DIR}/${PREFIX}.mt.prepped ${DIR}/${PREFIX}.mt.prepped.orig 
        cp ${DIR}/${PREFIX}.pe.prepped ${DIR}/${PREFIX}.pe.prepped.orig
        
        COUNTER=0
        for LINE_NO in ${DELETE_LINES[@]}
        do 
            ACTUAL_LINE_NO=`expr $LINE_NO - $COUNTER`
            echo "Deleting line: $ACTUAL_LINE_NO"
            sed  -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.src.prepped
            sed -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.mt.prepped
            sed -i.bak -e "${ACTUAL_LINE_NO}d" ${DIR}/${PREFIX}.pe.prepped
            COUNTER=$((COUNTER + 1))
        done

        python $qe_seq_dir/scripts/create_factor_corpus.py --f1 ${DIR}/${PREFIX}.mt.prepped --f2 ${DIR}/${PREFIX}.mt_source_aligned --output ${DIR}/${PREFIX}.mt.factor_corpus

    done

done

