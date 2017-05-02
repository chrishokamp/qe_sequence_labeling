#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=~/projects/nematus/

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder/

# theano device, in case you do not want to compute on gpu, change it to cpu
device=cuda0

#model prefix
prefix=model/model.npz

DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017
dev=$DATADIR/dev.mt.factor_corpus
ref=$DATADIR/dev.pe

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $dev.output.fine_tune.dev \
     -k 12 -n -p 1


bash postprocess-dev.sh < $dev.output.fine_tune.dev > $dev.output.fine_tune.postprocessed.dev

## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.fine_tune.postprocessed.dev >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.fine_tune.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

## get f1 product using TER alignment
DROPBOX=/media/1tb_drive/Dropbox
TERCOM=$DROPBOX/data/qe/sw/tercom-0.7.25
# note we use the APE hypothesis as the pseudo-ref
pseudo_ref=$dev.output.fine_tune.postprocessed.dev
orig_hyps=$DATADIR/dev.mt
SRC_LANG=en
TRG_LANG=de
TMP_DIR=dev_ter_tmp
mkdir -p $TMP_DIR

qe_sequence_labeling=~/projects/qe_sequence_labeling/

TIMESTAMP=`date +"%Y-%m-%d_%H-%M-%S"`


python $qe_sequence_labeling/scripts/qe_labels_from_ter_alignment.py --hyps $orig_hyps --refs $pseudo_ref --output $TMP_DIR --src_lang $SRC_LANG --trg_lang $TRG_LANG --tercom $TERCOM

# now compute F1 product from two files
python $qe_sequence_labeling/scripts/qe_metrics_from_files.py --hyps $TMP_DIR/${SRC_LANG}-${TRG_LANG}.tercom.out.tags --gold $DATADIR/dev.tags --output $TMP_DIR/qe_dev_report_${TIMESTAMP}

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi
