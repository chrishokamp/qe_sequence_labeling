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

# extract all factor vocabularies after concatenation but before rejoining with text corpora
Concatenate all source and target text and factor corpora
```
SEQ_QE=~/projects/qe_sequence_labeling

# Concat text factors
bash $SEQ_QE/experiment_configs/nematus/concat/de-en/concat_text_factors.sh

# Note this script has the paths to 4M, 500K, and APE_INTERNAL hard-coded inside it
# Concat Spacy factors
bash $SEQ_QE/experiment_configs/nematus/concat/de-en/concat_factor_corpora.sh
```


# extract factor vocabularies -- could write to tmp file for each factor and use nematus script
```
SEQ_QE=~/projects/qe_sequence_labeling
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus

python $SEQ_QE/scripts/extract_factor_vocabularies.py -i $DATADIR/train.src-mt.concatenated.factors.bpe -o $DATADIR -n 3
```

Map text and factor tokens back together
```
SEQUENCE_QE=~/projects/qe_sequence_labeling

# 4M
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.src-mt.concatenated.bpe -f $DATADIR/train.src-mt.concatenated.factors.bpe

# 500K
DATADIR=/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K/spacy_factor_corpus
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.src-mt.concatenated.bpe -f $DATADIR/train.src-mt.concatenated.factors.bpe

# APE Internal
DATADIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus

# Train
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/train.src-mt.concatenated.bpe -f $DATADIR/train.src-mt.concatenated.factors.bpe

# Dev
python $SEQUENCE_QE/scripts/join_text_with_factor_corpus.py -t $DATADIR/dev.src-mt.concatenated.bpe -f $DATADIR/dev.src-mt.concatenated.factors.bpe
```

NOTE: For the Spacy factored datasets, remember that we need the version of reference data without rows removed
Note: we move *.orig *pe files to spacy corpus 500K and APE Internal dirs


### Ensemble Decoding for APE
Decode with a set of models, which may each have different inputs
```
export SRC_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
export MT_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe
export DEV_DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017
export GBS_DIR=/home/chris/projects/constrained_decoding

src-pe + mt-pe
python $GBS_DIR/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.iter370000.npz $MT_MODEL_DIR/model.iter290000.npz \
 -c $SRC_MODEL_DIR/model.iter370000.npz.json $MT_MODEL_DIR/model.iter290000.npz.json \
 -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped > test.nematus.out


export MT_WITH_SRC_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de/fine_tune/model

# src-->pe + mt-->pe + mt-src-->pe
python $GBS_DIR/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.iter370000.npz $MT_MODEL_DIR/model.iter290000.npz $MT_WITH_SRC_MODEL_DIR/model.npz \
 -c $SRC_MODEL_DIR/model.iter370000.npz.json $MT_MODEL_DIR/model.iter290000.npz.json $MT_WITH_SRC_MODEL_DIR/model.npz.json \
 -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped $DEV_DATA_DIR/dev.mt.factor_corpus > test.src_mt_ens.nematus.out

```

## Tuning

# WORKING: note that we _know_ which model is the best from each of the runs -- check perf with i.e. averaging `best_model` 
# WORKING: from each of the run types

#### Optimize Ensemble Decoding Weights with MERT

Get the paths to the best N models in a directory
```
CONSTRAINED_DECODING=~/projects/constrained_decoding
MODEL_BASEDIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models

# src-mt_concat 
EXP_DIR=$MODEL_BASEDIR/en-de_concat_src_mt/fine_tune/min_risk/model
python $CONSTRAINED_DECODING/scripts/get_best_n_nematus_models.py -m $EXP_DIR -k 5

# mt-src_aligned 
EXP_DIR=$MODEL_BASEDIR/en-de_mt_aligned/fine_tune/model
python $CONSTRAINED_DECODING/scripts/get_best_n_nematus_models.py -m $EXP_DIR -k 5

# src-mt_concat factors 
EXP_DIR=$MODEL_BASEDIR/en-de_concat_factors/fine_tune/min_risk/model
python $CONSTRAINED_DECODING/scripts/get_best_n_nematus_models.py -m $EXP_DIR -k 5

```

SRC-MT CONCAT
BEST MODELS

/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model/model.iter52000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model/model.iter30000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model/model.iter58000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model/model.iter50000.npz

BEST SCORES

69.44
69.37
69.34
69.32

MT-SRC ALIGNED
--------------
BEST MODELS

/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model/model.iter42000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model/model.iter32000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model/model.iter3000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model/model.iter48000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model/model.iter27000.npz

BEST SCORES

67.54
67.51
67.49
67.47
67.44

SRC-MT CONCAT FACTORS
----------------------

BEST MODELS

/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model/model.iter24000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model/model.iter2000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model/model.iter78000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model/model.iter79000.npz
/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model/model.iter69000.npz

BEST SCORES

69.76
69.76
69.68
69.65
69.65


# WORKING -- final ensemble prep flowchart (single, ensemble, tuned ensemble) x (ape,qe)



### Evaluation

Evaluate an APE system on WMT 16 dev and WMT 16 test for each of {APE,QE} -- this requires four translation passes 
If the model is an ensemble, weights should be different between APE and QE runs
```
# set environment vars

HYPS=


```

Average N-models, and output a new model
```
GBS=~/projects/constrained_decoding

# src-pe
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
python $GBS/scripts/average_nematus_models.py -m $MODEL_DIR/model.iter340000.npz $MODEL_DIR/model.iter350000.npz $MODEL_DIR/model.iter360000.npz $MODEL_DIR/model.iter370000.npz -o $MODEL_DIR/model.4-best.averaged.npz

# mt-pe
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe
python $GBS/scripts/average_nematus_models.py -m $MODEL_DIR/model.iter260000.npz $MODEL_DIR/model.iter270000.npz $MODEL_DIR/model.iter280000.npz $MODEL_DIR/model.iter290000.npz -o $MODEL_DIR/model.4-best.averaged.npz

# mt-src_aligned
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model
python $GBS/scripts/average_nematus_models.py -m $MODEL_DIR/model.iter42000.npz $MODEL_DIR/model.iter32000.npz $MODEL_DIR/model.iter3000.npz $MODEL_DIR/model.iter48000.npz -o $MODEL_DIR/model.4-best.averaged.npz

# concat_src-mt_factors
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model
python $GBS/scripts/average_nematus_models.py -m $MODEL_DIR/model.iter24000.npz $MODEL_DIR/model.iter2000.npz $MODEL_DIR/model.iter78000.npz $MODEL_DIR/model.iter79000.npz -o $MODEL_DIR/model.4-best.averaged.npz

# concat_src-mt
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model
python $GBS/scripts/average_nematus_models.py -m $MODEL_DIR/model.iter52000.npz $MODEL_DIR/model.iter30000.npz $MODEL_DIR/model.iter58000.npz $MODEL_DIR/model.iter50000.npz -o $MODEL_DIR/model.4-best.averaged.npz

```

SRC-PE
```
# WORKING script to translate with model args -- i.e. eval setup in script?
GBS=~/projects/constrained_decoding
QE_SEQ=~/projects/qe_sequence_labeling/
MOSES_SCRIPTS=~/projects/mosesdecoder/scripts
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

# All models averaged
MODEL_0=$MODEL_DIR/model.4-best.averaged.npz

# Note there will be multiple models for ensembles
MODEL_1=$MODEL_DIR/model.iter340000.npz
MODEL_2=$MODEL_DIR/model.iter350000.npz
MODEL_3=$MODEL_DIR/model.iter360000.npz
MODEL_4=$MODEL_DIR/model.iter370000.npz

# Note there will be weights for tuned ensembles
# TODO: is this dev from QE or APE -- this is important for test output 
# TODO: the difference between QE and APE data is probably only important for test outputs, not for tuning/finding best model
INPUT=$DATA_DIR/dev.src.prepped
MT=$DATA_DIR/dev.mt
REF=$DATA_DIR/dev.pe
TAGS=$DATA_DIR/dev.tags

OUTPUT_DIR=/media/1tb_drive/nematus_ape_experiments/evaluation_results/src-pe

mkdir -p $OUTPUT_DIR

# translate
# Single SRC model
SINGLE_OUTPUT_FILE=$OUTPUT_DIR/dev.output.postprocessed
python $GBS/scripts/translate_nematus.py -m $MODEL_0 -c $MODEL_DIR/model.npz.json -i $INPUT | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $SINGLE_OUTPUT_FILE

# Ensemble of 4 SRC models
ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/dev-ensemble-4.output.postprocessed
python $GBS/scripts/translate_nematus.py -m $MODEL_1 $MODEL_2 $MODEL_3 $MODEL_4 -c $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json -i $INPUT $INPUT $INPUT $INPUT | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $SINGLE_OUTPUT_FILE
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $ENSEMBLE_OUTPUT_FILE

```

MT-PE

```
GBS=~/projects/constrained_decoding
QE_SEQ=~/projects/qe_sequence_labeling/
MOSES_SCRIPTS=~/projects/mosesdecoder/scripts
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

# All models averaged
MODEL_0=$MODEL_DIR/model.4-best.averaged.npz

# Note there will be multiple models for ensembles
MODEL_1=$MODEL_DIR/model.iter260000.npz
MODEL_2=$MODEL_DIR/model.iter270000.npz
MODEL_3=$MODEL_DIR/model.iter280000.npz
MODEL_4=$MODEL_DIR/model.iter290000.npz

INPUT=$DATA_DIR/dev.mt.prepped
MT=$DATA_DIR/dev.mt
REF=$DATA_DIR/dev.pe
TAGS=$DATA_DIR/dev.tags

OUTPUT_DIR=/media/1tb_drive/nematus_ape_experiments/evaluation_results/mt-pe
mkdir -p $OUTPUT_DIR

# translate
# Single model
SINGLE_OUTPUT_FILE=$OUTPUT_DIR/dev.output.postprocessed
python $GBS/scripts/translate_nematus.py -m $MODEL_0 -c $MODEL_DIR/model.npz.json -i $INPUT | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $SINGLE_OUTPUT_FILE

# Ensemble of 4 models
ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/dev-ensemble-4.output.postprocessed
python $GBS/scripts/translate_nematus.py -m $MODEL_1 $MODEL_2 $MODEL_3 $MODEL_4 -c $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json $MODEL_DIR/model.npz.json -i $INPUT $INPUT $INPUT $INPUT | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $SINGLE_OUTPUT_FILE
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $ENSEMBLE_OUTPUT_FILE

```


Concat Factors
```
# best model from training vs avg model for concat factors

GBS=~/projects/constrained_decoding
QE_SEQ=~/projects/qe_sequence_labeling/
MOSES_SCRIPTS=~/projects/mosesdecoder/scripts
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model

# All models averaged
MODEL_0=$MODEL_DIR/model.4-best.averaged.npz
BEST_MODEL=$MODEL_DIR/model.npz.npz.best_bleu

OUTPUT_DIR=/media/1tb_drive/nematus_ape_experiments/evaluation_results/concat_factors
mkdir -p $OUTPUT_DIR

# DEV
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017
INPUT=$DATA_DIR/spacy_factor_corpus/dev.src-mt.concatenated.bpe.factor_corpus
MT=$DATA_DIR/dev.mt
REF=$DATA_DIR/dev.pe
TAGS=$DATA_DIR/dev.tags

# translate
# Single model
SINGLE_OUTPUT_FILE=$OUTPUT_DIR/dev.output.postprocessed.4-best.averaged
python $GBS/scripts/translate_nematus.py -m $MODEL_0 -c $MODEL_DIR/model.npz.json -i $INPUT --length_factor 1.0 --beam_size 5 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $SINGLE_OUTPUT_FILE

# Best training model 
BEST_OUTPUT_FILE=$OUTPUT_DIR/dev.output.postprocessed.best-training-model
python $GBS/scripts/translate_nematus.py -m $BEST_MODEL -c $MODEL_DIR/model.npz.json -i $INPUT --length_factor 1.0 --beam_size 5 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $BEST_OUTPUT_FILE

# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $SINGLE_OUTPUT_FILE
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $BEST_OUTPUT_FILE

# TEST
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2016
INPUT=$DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus
MT=$DATA_DIR/test.mt
REF=$DATA_DIR/test.pe
TAGS=$DATA_DIR/test.tags

# translate
# Single model
SINGLE_OUTPUT_FILE=$OUTPUT_DIR/test.output.postprocessed.4-best.averaged
python $GBS/scripts/translate_nematus.py -m $MODEL_0 -c $MODEL_DIR/model.npz.json -i $INPUT --length_factor 1.0 --beam_size 5 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $SINGLE_OUTPUT_FILE

# Best training model 
BEST_OUTPUT_FILE=$OUTPUT_DIR/test.output.postprocessed.best-training-model
python $GBS/scripts/translate_nematus.py -m $BEST_MODEL -c $MODEL_DIR/model.npz.json -i $INPUT --length_factor 1.0 --beam_size 5 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $BEST_OUTPUT_FILE

# Evaluate
# TODO: eval script must be updated to include test paths -- make sure to differentiate between QE and APE *.pe inputs
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m test -h $SINGLE_OUTPUT_FILE
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m test -h $BEST_OUTPUT_FILE

```

AVG_ALL Baseline Ensemble
```
GBS=~/projects/constrained_decoding
QE_SEQ=~/projects/qe_sequence_labeling/
MOSES_SCRIPTS=~/projects/mosesdecoder/scripts
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe

SRC_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
MT_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe
MT_ALIGN_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model
CONCAT_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model
CONCAT_FACTORS_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model

OUTPUT_DIR=/media/1tb_drive/nematus_ape_experiments/evaluation_results/avg_all_ensemble
mkdir -p $OUTPUT_DIR

# DEV
DEV_DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/dev.ensemble.output.postprocessed

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped $DEV_DATA_DIR/dev.mt.factor_corpus $DEV_DATA_DIR/dev.src-mt.concatenated $DEV_DATA_DIR/spacy_factor_corpus/dev.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $ENSEMBLE_OUTPUT_FILE

# 2016 Test
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2016

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/test.ensemble.output.postprocessed

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DATA_DIR/test.src.prepped $DATA_DIR/test.mt.prepped $DATA_DIR/test.mt.factor_corpus $DATA_DIR/test.src-mt.concatenated $DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m test -h $ENSEMBLE_OUTPUT_FILE

# WMT 2017 Test -- Note we assume that QE_input == APE_input 
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2017

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/test_2017.ensemble.output.postprocessed

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DATA_DIR/test.src.prepped $DATA_DIR/test.mt.prepped $DATA_DIR/test.mt.factor_corpus $DATA_DIR/test.src-mt.concatenated $DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

```

AVG_ALL Tuned Ensemble
```
GBS=~/projects/constrained_decoding
QE_SEQ=~/projects/qe_sequence_labeling/
MOSES_SCRIPTS=~/projects/mosesdecoder/scripts
MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe

SRC_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe
MT_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe
MT_ALIGN_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model
CONCAT_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_src_mt/fine_tune/min_risk/model
CONCAT_FACTORS_MODEL_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_concat_factors/fine_tune/min_risk/model

OUTPUT_DIR=/media/1tb_drive/nematus_ape_experiments/evaluation_results/avg_all_ensemble
mkdir -p $OUTPUT_DIR

# weights from tuning
OPTIMIZATION_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/mert_optimization/avg_all/tuning.1494628152
WEIGHTS=$OPTIMIZATION_DIR/run10.dense

# DEV
DEV_DATA_DIR=/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/dev.ensemble.output.postprocessed.tuned

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped $DEV_DATA_DIR/dev.mt.factor_corpus $DEV_DATA_DIR/dev.src-mt.concatenated $DEV_DATA_DIR/spacy_factor_corpus/dev.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 --load_weights $WEIGHTS | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE
# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m dev -h $ENSEMBLE_OUTPUT_FILE

# Test
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2016

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/test.ensemble.output.postprocessed.tuned

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DATA_DIR/test.src.prepped $DATA_DIR/test.mt.prepped $DATA_DIR/test.mt.factor_corpus $DATA_DIR/test.src-mt.concatenated $DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 --load_weights $WEIGHTS | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE
# Evaluate
bash $QE_SEQ/scripts/evaluate_ape_and_qe.sh -m test -h $ENSEMBLE_OUTPUT_FILE

# WMT 2017 Test -- Note we assume that QE_input == APE_input 
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2017

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/test_2017.ensemble.output.postprocessed.tuned

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DATA_DIR/test.src.prepped $DATA_DIR/test.mt.prepped $DATA_DIR/test.mt.factor_corpus $DATA_DIR/test.src-mt.concatenated $DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 --load_weights $WEIGHTS | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE


# QE Tuned Ensemble

# weights from tuning
# with run4 weights
OPTIMIZATION_DIR=/media/1tb_drive/nematus_ape_experiments/ape_qe/mert_optimization/avg_all_f1_product/tuning.1494697732
WEIGHTS=$OPTIMIZATION_DIR/run4.dense

# dev
ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/dev.ensemble.output.postprocessed.run4_qe_tuned

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped $DEV_DATA_DIR/dev.mt.factor_corpus $DEV_DATA_DIR/dev.src-mt.concatenated $DEV_DATA_DIR/spacy_factor_corpus/dev.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 --load_weights $WEIGHTS | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE

# test
DATA_DIR=/media/1tb_drive/Dropbox/data/qe/wmt_2017/test/wmt17_qe_test_data/word_level/2016

ENSEMBLE_OUTPUT_FILE=$OUTPUT_DIR/test.ensemble.output.postprocessed.tuned.run4_qe_tuned

python $GBS/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.4-best.averaged.npz $MT_MODEL_DIR/model.4-best.averaged.npz $MT_ALIGN_MODEL_DIR/model.4-best.averaged.npz $CONCAT_MODEL_DIR/model.4-best.averaged.npz $CONCAT_FACTORS_MODEL_DIR/model.4-best.averaged.npz -c $SRC_MODEL_DIR/model.npz.json $MT_MODEL_DIR/model.npz.json $MT_ALIGN_MODEL_DIR/model.npz.json $CONCAT_MODEL_DIR/model.npz.json $CONCAT_FACTORS_MODEL_DIR/model.npz.json -i $DATA_DIR/test.src.prepped $DATA_DIR/test.mt.prepped $DATA_DIR/test.mt.factor_corpus $DATA_DIR/test.src-mt.concatenated $DATA_DIR/spacy_factor_corpus/test.src-mt.concatenated.bpe.factor_corpus --beam_size 5 --length_factor 2.0 --load_weights $WEIGHTS | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $ENSEMBLE_OUTPUT_FILE


```

MERT N-best output to 1-best list -- for sanity evaluation of tuning passes -- note detruecasing should be done after this
```
cat run4.out | perl -ne 'chomp; @t = split(/\|\|\|/, $_); print "$t[1]\n"' | sed -n '1~10p' > run4.1best.out

```


Prepare APE submission
```
~/projects/qe_sequence_labeling/scripts/prepare_ape_wmt_submission.py
```



TODO: QE Avg All -- F1 product tuned ensemble


=======
### Notes

We also want to try:
(1) min-risk training (ideally directly against TER)
(2) constrained decoding with terminology extracted from in-domain data
    - the terms should be extracted from (mt-pe), and should be errors which are _always_ corrected when we see them

See here: https://explosion.ai/blog/german-model#word-order for more info on the german dependency parsing model

Concatenation could also add special <SRC> and <TRG> tokens to make explicit which language sequence follows
  the intuition here is that, since the segmentation is learned jointly, we want to be explicit about which language
  we are currently using

Remember that tagsets are across both languages for the concatenated models. The <JOIN> token also needs to be duplicated across factors

Note: Text factor VOCAB EXTRACTION IS DONE IN THE CONCATENATION SCRIPT

#### Dev
Note we use the QE data as development data -- this will allow us to also compute TER, and score for both QE and APE during validation


