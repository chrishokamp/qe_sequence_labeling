#!/bin/bash -v

APE_OUTPUT_FILE_DEV=$1
APE_OUTPUT_FILE_TEST=$2

python run_tercom.py \
    -target_file=dev.mt \
    -post_edited_file=${APE_OUTPUT_FILE_DEV} >& /dev/null
python parse_pra.py \
    -tercom_file=${APE_OUTPUT_FILE_DEV}.tercom.out.pra
python generate_wmt_submission.py \
    AMU \
    dev.mt \
    ${APE_OUTPUT_FILE_DEV}.tercom.out.pra.tags \
    > ${APE_OUTPUT_FILE_DEV}.qe.submission
python evaluate_wmt15.py \
    dev.mt \
    dev.tags \
    ${APE_OUTPUT_FILE_DEV}.qe.submission
rm ${APE_OUTPUT_FILE_DEV}.tercom.*