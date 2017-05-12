#!/usr/bin/env bash

usage() { echo "Usage: $0 [-m <one of {dev,test}>] [-h <translation-hyps>]" 1>&2; exit 1; }

die() { echo "$@" 1>&2 ; exit 1; }

while getopts ":m:h:" o; do
    case "${o}" in
        m)
            mode=${OPTARG}
            ;;
        h)
            hyp=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${mode}" ] || [ -z "${hyp}" ]; then
    usage
fi

echo "hyp file = ${hyp}"
echo "mode = ${mode}"

MOSES_SCRIPTS=~/projects/mosesdecoder/scripts

# WORKING: arg for dev|test
if [ ${mode} == 'dev' ]; then
    DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017
else
    die "Unknown mode: $mode"
fi

# Note there will be weights for tuned ensembles
# TODO: is this dev from QE or APE -- this is important for test output
# TODO: the difference between QE and APE data is probably only important for test outputs, not for tuning/finding best model

OUTPUT_FILE=${hyp}
MT=$DATA_DIR/dev.mt
REF=$DATA_DIR/dev.pe
TAGS=$DATA_DIR/dev.tags

# Take translation output and evaluate against WMT 2016 dev or test for APE and QE metrics
# BLEU Score
BLEU=`$MOSES_SCRIPTS/generic/multi-bleu.perl $REF < $OUTPUT_FILE | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "BLEU = $BLEU"

DROPBOX=/media/1tb_drive/Dropbox
TERCOM=$DROPBOX/data/qe/sw/tercom-0.7.25
# note we use the APE hypothesis as the pseudo-ref
pseudo_ref=$dev.output.postprocessed.dev
orig_hyps=$DATADIR/dev.mt
SRC_LANG=en
TRG_LANG=de
TMP_DIR=dev_ter_tmp
mkdir -p $TMP_DIR

# TODO: add official QE evaluation script?
printf "Quality Estimation Metrics\n"
QE_SEQ=~/projects/qe_sequence_labeling/
TIMESTAMP=`date +"%Y-%m-%d_%H-%M-%S"`
python $QE_SEQ/scripts/qe_labels_from_ter_alignment.py --hyps $MT --refs $OUTPUT_FILE --output $TMP_DIR --src_lang $SRC_LANG --trg_lang $TRG_LANG --tercom $TERCOM > /dev/null 2>&1

# now compute F1 product from two files
python $QE_SEQ/scripts/qe_metrics_from_files.py --hyps $TMP_DIR/${SRC_LANG}-${TRG_LANG}.tercom.out.tags --gold $TAGS --output $TMP_DIR/qe_dev_report_${TIMESTAMP} > /dev/null 2>&1
cat $TMP_DIR/qe_dev_report_${TIMESTAMP}.json

# now run WMT APE evaluation script
# evaluate APE
printf "\nAPE TER\n"
bash $QE_SEQ/scripts/wmt_ape_evaluation/Evaluation_Script/runTER.sh -h $OUTPUT_FILE -r $REF -s $TIMESTAMP -o $TMP_DIR > $TMP_DIR/ape_ter_output
cat $TMP_DIR/ape_ter_output | grep 'TER'

# Cleanup
rm -rf $TMP_DIR
rm *postprocessed*ter
rm *tercom*
