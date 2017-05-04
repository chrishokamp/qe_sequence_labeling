Training an APE system using factored NMT

Train EN-DE and DE-EN baseline systems using Nematus
- we'll use these systems to generate the alignment features used for factored input to the APE system.

#### prepare data

We start from the seq2seq EN-DE corpus (already tokenized and BPE segmented)

```
# Build dictionaries in Nematus format
export DATA_DIR=/media/1tb_drive/parallel_data/en-de/google_seq2seq_dataset
export NEMATUS=~/projects/nematus
export SRC_TRAIN=$DATA_DIR/train.tok.clean.bpe.32000.en
export TRG_TRAIN=$DATA_DIR/train.tok.clean.bpe.32000.de
$NEMATUS/data/build_dictionary.py $SRC_TRAIN $TRG_TRAIN
```

#### Make Nematus config files 

Each language pair needs its own `config.py`, `validate.sh`, `postprocess-dev.sh`, and `postprocess-test.sh` files.
The templates for these files can be found in `wmt16-scripts/sample`

Following the Nematus convention, we place these into the directory where the model and training logs should be saved

Note: Nematus shuffles data in `/tmp`, if your devices runs out of space, you may need to do `rm /tmp/tmp*`

##### Train EN-DE system
```
source activate theano
export CONFIG_DIR=~/projects/qe_sequence_labeling/experiment_configs/nematus/train/en-de
export MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/nmt_baselines/en-de
mkdir -p $MODEL_DIR
mkdir -p $MODEL_DIR/model
cp $CONFIG_DIR/* $MODEL_DIR

cd $MODEL_DIR
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda,on_unused_input=warn python config.py

```



TODO(chrishokamp): the workflow below has been superceded by the scripts in `experiment_configs/nematus/preprocess`
TODO(chrishokamp): remove most of the specific commands below and refer reader to the scripts


#### Forced decoding and extracting alignments 

Add segment numbers to the MT to make the file appear to be an n-best list
```
awk '{ print FNR - 1 " ||| " $0 }' train.mt > train.mt.numbered
```

Use `experiment_configs/nematus/rescore/<src>-<trg>/rescore.sh` to get the alignment weights between SOURCE and MT
This script preprocesses the data, then calls `nematus/nematus/rescore.py` to do the forced alignment.

WORKING: add preprocess.sh, factor out from rescore.sh
WORKING: remember that QE data must be prepped in exactly the same way as MT data

EN-DE
```
source activate theano
export RESCORE_SCRIPT=~/projects/qe_sequence_labeling/experiment_configs/nematus/rescore/en-de/rescore.sh
export MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de
cp $RESCORE_SCRIPT $MODEL_DIR
cd $MODEL_DIR
bash rescore.sh
```

DE-EN
```
source activate theano
export RESCORE_SCRIPT=~/projects/qe_sequence_labeling/experiment_configs/nematus/rescore/de-en/rescore.sh
export MODEL_DIR=/extra/chokamp/nmt_systems/pretrained_wmt16_models/de-en
cp $RESCORE_SCRIPT $MODEL_DIR
cd $MODEL_DIR
bash rescore.sh
```

#### Extract aligned word factors

```
export DATA_DIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16/
python scripts/alignment_corpus_from_nematus_json_output.py --json $DATA_DIR/train.mt.rescored_withwords.json --output $DATA_DIR/train.mt.aligned_words.target_order.factor --order target
```

IMPORTANT: The Nematus json output is messed up. There are hacks to get around this in
`scripts/alignment_corpus_from_nematus_json_output.py`, but there can still be rows that
we cannot parse. When there is an unparsable row, the script will output its index in the corpus
This index _must_ be deleted from the corresponding `.src, .mt, .pe, .tags` files, otherwise the 
datasets will no longer be sentence-aligned.
```
ERROR:__main__:DELETED ROW: 386771
```

```
export FACTOR_CORPUS=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16/factored_ape_corpus
cd $FACTOR_CORPUS

sed -i.bak -e '386771d' train.rescore.preprocessed.src
sed -i.bak -e '386771d' train.rescore.preprocessed.mt
```

#### Extract Aligned Word Factors from very large artificial data

Since we use the WMT pretrained models, we need to prep the dev corpus accordingly
Use WMT 16 QE dev
```
export SUBWORD_NMT=~/projects/subword_nmt
export BPE_CODES=/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de/ende.bpe
export QE_DATA=/media/1tb_drive/Dropbox/data/qe/wmt_2016/dev_wmt16_pretrained_bpe

# map dev through subword
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev.src > $QE_DATA/dev.src.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev.mt > $QE_DATA/dev.mt.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev.pe > $QE_DATA/dev.pe.bpe

# now extract factors for dev
source activate theano
export RESCORE_SCRIPT=~/projects/qe_sequence_labeling/experiment_configs/nematus/rescore/en-de/rescore_wmt16_qe_dev.sh
export MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de
cp $RESCORE_SCRIPT $MODEL_DIR
cd $MODEL_DIR
bash $RESCORE_SCRIPT

# now extract alignments
cd ~/projects/qe_sequence_labeling
python scripts/alignment_corpus_from_nematus_json_output.py --json $QE_DATA/dev.mt.rescored_withwords.json --output $QE_DATA/dev.mt.aligned_words.target_order.factor --order target
```

#### Map train *.pe through pretrained model bpe (remember to remove extra lines that are broken from processing Nematus json)
```
export LANG=de
export ORIG_DATA_DIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16
export OUTPUT_DIR=$ORIG_DATA_DIR/factored_ape_corpus
export BPE_CODES=/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de/ende.bpe
export mosesdecoder=~/projects/mosesdecoder
export subword_nmt=~/projects/subword_nmt

# $mosesdecoder/scripts/tokenizer/escape-special-chars.perl -l $LANG | \

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $LANG | \ 
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $LANG -penn | \ 
$subword_nmt/apply_bpe.py -c $BPE_CODES < $ORIG_DATA_DIR/train.pe > $OUTPUT_DIR/train.pe.prepped

# Remove the broken line from PE references
sed -i.bak -e '386771d' $OUTPUT_DIR/train.pe.prepped
```

#### Map dev *.pe through pretrained model bpe
```
export LANG=de
export DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2016/dev_wmt16_pretrained_bpe
export BPE_CODES=/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de/ende.bpe
export mosesdecoder=~/projects/mosesdecoder
export subword_nmt=~/projects/subword_nmt

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $LANG | $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 10 -l $LANG -penn | $subword_nmt/apply_bpe.py -c $BPE_CODES < $DATA_DIR/dev.pe > $DATA_DIR/dev.pe.prepped
```

#### Create corpus for factors, using the Nematus pipe separator

#### 500k+WMT QE Train
```
# Train
export FACTOR_CORPUS=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16/factored_ape_corpus
python scripts/create_factor_corpus.py --f1 $FACTOR_CORPUS/train.rescore.preprocessed.mt --f2 $FACTOR_CORPUS/train.mt.aligned_words.target_order.factor --output $FACTOR_CORPUS/train.mt_aligned_with_source.factor

# Dev
export FACTOR_CORPUS=/media/1tb_drive/Dropbox/data/qe/wmt_2016/dev_wmt16_pretrained_bpe
python scripts/create_factor_corpus.py --f1 $FACTOR_CORPUS/dev.mt.bpe.prepped --f2 $FACTOR_CORPUS/dev.mt.aligned_words.target_order.factor --output $FACTOR_CORPUS/dev.mt_aligned_with_source.factor

```

Train factored system on factored 500k+WMT QE APE corpus, validate with BLEU + F1_Product on factored QE data
```
source activate theano
export CONFIG_DIR=~/projects/qe_sequence_labeling/experiment_configs/nematus/train/ape/en-de_factored
export MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de
mkdir -p $MODEL_DIR
mkdir -p $MODEL_DIR/model
cp $CONFIG_DIR/* $MODEL_DIR

cd $MODEL_DIR
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda,on_unused_input=warn python config.py

```

Train SRC-->PE and MT-->PE baselines on 500k+WMT QE APE corpus

Dev process (i.e. the job of `validate.sh`):
(1) translate BPE segmented dev data
(2) unsegment translated output
(3) compute TER tags per-line for translated output against dev.pe
(4) compute f1 product against gold tags, use as validation metric


### Creating a Factor Corpus with Spacy

Extract a concatenated source corpus with rich features (POS tags, head words, dependency relations)

```
source activate spacy 
QE_SEQ=~/projects/qe_sequence_labeling

# 4M
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M
# src
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.src -o $DATADIR/spacy_factor_corpus -l en -p train
# mt 
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.mt -o $DATADIR/spacy_factor_corpus -l de -p train

# 500K
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K
# src
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.src -o $DATADIR/spacy_factor_corpus -l en -p train
# mt 
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.mt -o $DATADIR/spacy_factor_corpus -l de -p train

# APE internal
DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017
# train
# src
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.src -o $DATADIR/spacy_factor_corpus -l en -p train
# mt 
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/train.mt -o $DATADIR/spacy_factor_corpus -l de -p train
# dev
# src
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/dev.src -o $DATADIR/spacy_factor_corpus -l en -p dev 
# mt 
python $QE_SEQ/scripts/generate_factor_corpus_with_spacy.py -i $DATADIR/dev.mt -o $DATADIR/spacy_factor_corpus -l de -p dev 

# APE test data

# QE test data
```

Learn joint BPE encoding for the text factor of the concatenated Spacy extracted corpus
INFO:__main__:Wrote new files: /media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus/train.en.tok and /media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus/train.en.factors
```
# follow subword-nmt best practices here
SUBWORD=~/projects/subword_nmt
NUM_OPERATIONS=40000

# Learn BPE on 4M dataset
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus

python $SUBWORD/learn_joint_bpe_and_vocab.py --input $DATADIR/train.en.tok $DATADIR/train.de.tok -s $NUM_OPERATIONS -o $DATADIR/en-de.spacy_tokenization.codes.bpe --write-vocabulary $DATADIR/vocab.en $DATADIR/vocab.de 

# Apply the vocabulary to the train data with the vocabulary threshold enabled
python $SUBWORD/apply_bpe.py -c $DATADIR/en-de.spacy_tokenization.codes.bpe --vocabulary $DATADIR/vocab.en --vocabulary-threshold 50 < $DATADIR/train.en.tok > $DATADIR/train.en.bpe
python $SUBWORD/apply_bpe.py -c $DATADIR/en-de.spacy_tokenization.codes.bpe --vocabulary $DATADIR/vocab.de --vocabulary-threshold 50 < $DATADIR/train.de.tok > $DATADIR/train.de.bpe

# for other data, re-use the same options for consistency:
# 500K
VOCAB_DIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K/spacy_factor_corpus

python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.en --vocabulary-threshold 50 < $DATADIR/train.en.tok > $DATADIR/train.en.bpe
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.de --vocabulary-threshold 50 < $DATADIR/train.de.tok > $DATADIR/train.de.bpe

# APE Internal
VOCAB_DIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus

# train
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.en --vocabulary-threshold 50 < $DATADIR/train.en.tok > $DATADIR/train.en.bpe
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.de --vocabulary-threshold 50 < $DATADIR/train.de.tok > $DATADIR/train.de.bpe

# dev
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.en --vocabulary-threshold 50 < $DATADIR/dev.en.tok > $DATADIR/dev.en.bpe
python $SUBWORD/apply_bpe.py -c $VOCAB_DIR/en-de.spacy_tokenization.codes.bpe --vocabulary $VOCAB_DIR/vocab.de --vocabulary-threshold 50 < $DATADIR/dev.de.tok > $DATADIR/dev.de.bpe

```

Map factors through BPE segmentation, add B-* and I-* factors
```
SEQUENCE_QE=~/projects/qe_sequence_labeling

# 4M
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
# src
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors
# mt
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors

# 500K
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K/spacy_factor_corpus
# src
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors
# mt
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors


# APE Internal
DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus

# Train
# src
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors
# mt
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors

# Dev
# src
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/dev.en.bpe -f $DATADIR/dev.en.factors
# mt
python $SEQUENCE_QE/scripts/map_factor_corpus_to_subword_segmentation.py -t $DATADIR/dev.de.bpe -f $DATADIR/dev.de.factors
```

# WORKING HERE
# TODO: extract all factor vocabularies after concatenation but before rejoining with text corpora
Concatenate all source and target text and factor corpora
```
SEQ_QE=~/projects/qe_sequence_labeling

# Concat text factors
bash $SEQ_QE/experiment_configs/nematus/concat/de-en/concat_text_factors.sh

# Note this script has the paths to 4M, 500K, and APE_INTERNAL hard-coded inside it
# Concat Spacy factors
bash $SEQ_QE/experiment_configs/nematus/concat/de-en/concat_factor_corpora.sh
```

Note: Text factor VOCAB EXTRACTION IS DONE IN THE CONCATENATION SCRIPT

Working here: extract factor vocabularies -- could write to tmp file for each factor and use nematus script

Map text and factor tokens back together
```
SEQUENCE_QE=~/projects/qe_sequence_labeling

# 4M
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
# src
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors.bpe
# mt
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors.bpe

# 500K
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K/spacy_factor_corpus
# src
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors.bpe
# mt
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors.bpe

# APE Internal
DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus

# Train
# src
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.en.bpe -f $DATADIR/train.en.factors.bpe
# mt
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.de.bpe -f $DATADIR/train.de.factors.bpe

# Dev
# src
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/dev.en.bpe -f $DATADIR/dev.en.factors.bpe
# mt
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/dev.de.bpe -f $DATADIR/dev.de.factors.bpe


```


TODO: redo dictionary extraction for concatenated data
TODO: concatenate data _before_ doing any BPE splitting, etc...

NOTE: For the Spacy factored datasets, remember that we need the version of reference data without rows removed
TODO: move *.orig *pe files to spacy corpus 500K and APE Internal dirs


TODO: output vocabularies for all factor tagsets using the mapped factor corpus
TODO: remember that tagsets are across both languages for the concatenated models
TODO: i.e. the tagset dict needs to include both EN and DE POS tags to work
TODO: Concatenate first, then extract vocabularies
TODO: The <JOIN> token also needs to be duplicated across factors




### Notes

We also want to try:
(1) min-risk training (ideally directly against TER)
(2) constrained decoding with terminology extracted from in-domain data
    - the terms should be extracted from (mt-pe), and should be errors which are _always_ corrected when we see them

See here: https://explosion.ai/blog/german-model#word-order for more info on the german dependency parsing model

Concatenation could also add special <SRC> and <TRG> tokens to make explicit which language sequence follows
  the intuition here is that, since the segmentation is learned jointly, we want to be explicit about which language
  we are currently using

#### Dev
Note we use the QE data as development data -- this will allow us to also compute TER, and score for both QE and APE during validation












