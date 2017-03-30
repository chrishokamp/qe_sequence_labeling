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

# Now copy all of the configs to $MODEL_DIR, an


##### Train DE-EN system






### Using WMT 16 Pre-trained models



```
# Download the files for the language pair that you want to use


```

#### Forced decoding and extracting alignments 

Add segment numbers to the MT to make the file appear to be an n-best list
```
awk '{ print FNR - 1 " ||| " $0 }' train.mt > train.mt.numbered
```

Use `nematus/rescore.py` to get the alignment weights between SOURCE and MT

EN-DE
```
source activate theano
export RESCORE_SCRIPT=~/projects/qe_sequence_labeling/experiment_configs/nematus/rescore/rescore.sh
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




