### QE Sequence Labeling

This is v2 of a project examining neural sequence models for token-level translation quality estimation.

This project uses the [sequence library](https://github.com/google/seq2seq) to create, train, and evaluate models. 


#### Dataset Preparation

Download WMT 16 QE data
Download WMT 17 QE data

Apply BPE segmentation to data (see below)


Create vocabularies


##### Learning BPE segmentation
We use a large dataset to learn the BPE segmentation jointly for the 
language pair.

```
export SUBWORD_NMT=/home/chris/projects/subword_nmt
export BPE_CODES=/media/1tb_drive/parallel_data/en-de/chris_en-de_big_corpus/train/bpe_20000_corpus/all_text_both_EN_and_DE.20000.bpe.codes
export QE_DATA=/media/1tb_drive/Dropbox/data/qe/wmt_2016

# map train
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/train/train.src > $QE_DATA/train/train.src.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/train/train.mt > $QE_DATA/train/train.mt.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/train/train.pe > $QE_DATA/train/train.pe.bpe

# map dev
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev/dev.src > $QE_DATA/dev/dev.src.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev/dev.mt > $QE_DATA/dev/dev.mt.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/dev/dev.pe > $QE_DATA/dev/dev.pe.bpe

# map test
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/test/test.src > $QE_DATA/test/test.src.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/test/test.mt > $QE_DATA/test/test.mt.bpe
python $SUBWORD_NMT/apply_bpe.py -c $BPE_CODES < $QE_DATA/test/test.pe > $QE_DATA/test/test.pe.bpe
```

Create the extended tag vocabularies, and map the tags to the subword segmentation of the MT hypotheses
```
export QE_DATA=/media/1tb_drive/Dropbox/data/qe/wmt_2016

# train
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/train/train.mt.bpe < $QE_DATA/train/train.tags > $QE_DATA/train/train.tags.mapped 
# dev 
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/dev/dev.mt.bpe < $QE_DATA/dev/dev.tags > $QE_DATA/dev/dev.tags.mapped 
# test
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/test/test.mt.bpe < $QE_DATA/test/test_words.tags > $QE_DATA/test/test.tags.mapped 
```

Create vocabulary dicts for source, target, and tags
```
# Create dicts for source and target 
export MT_DATA_DIR=/media/1tb_drive/parallel_data/en-de/chris_en-de_big_corpus/train/bpe_20000_corpus/
python scripts/create_vocabulary_indexes.py -s $MT_DATA_DIR/all_text.en-de.en.bpe.shuf -t $MT_DATA_DIR/all_text.en-de.de.bpe.shuf -v 25500 -o $MT_DATA_DIR -sl en -tl de

# INFO:__main__:Source Vocab size: 24273
# INFO:__main__:Target Vocab size: 24182

```

We can reuse vocabularies created for the same language pairs in MT experiments, for example. 
QE training datasets are pretty small, so reusing indices from larger datasets will probably generalize better to test data.
```
export MT_DATA_DIR=/media/1tb_drive/parallel_data/en-de/chris_en-de_big_corpus/train/bpe_20000_corpus/
export OUTPUT_DIR=/media/1tb_drive/Dropbox/data/qe/model_data/en-de

python scripts/map_vocab_index_to_qe_vocab.py -i $MT_DATA_DIR/de.vocab.pkl -o $OUTPUT_DIR/qe_output.vocab.pkl
# INFO:__main__:Wrote new index of size: 48361 to /media/1tb_drive/Dropbox/data/qe/model_data/en-de/qe_output.vocab.pkl

# copy the other indexes to our new directory
cp $MT_DATA_DIR/de.vocab.pkl $OUTPUT_DIR/de.vocab.pkl
cp $MT_DATA_DIR/en.vocab.pkl $OUTPUT_DIR/en.vocab.pkl
```

#### Recovering the original tag schema

Reduce fine-grained tags to {OK,BAD}
Evaluate using WMT F1 product evaluation script
 
### Running QE Experiments

#### Training a QE model

```
export DROPBOX_DIR=/home/chris/Desktop/Dropbox/
export QE_DATA_DIR=$DROPBOX_DIR/data/qe/wmt_2016
export RESOURCES=$DROPBOX_DIR/data/qe/model_data/en-de
export EXPERIMENT_DIR=/home/chris/projects/qe_sequence_labeling/experiments/test_unidirectional_qe

python scripts/train_qe_model.py -t $QE_DATA_DIR/train -v $QE_DATA_DIR/dev -l $EXPERIMENT_DIR -r $RESOURCES
```





