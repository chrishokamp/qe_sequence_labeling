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

Create the extended tag vocabularies
```
export QE_DATA=/media/1tb_drive/Dropbox/data/qe/wmt_2016

# train
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/train/train.mt.bpe < $QE_DATA/train/train.tags > $QE_DATA/train/train.tags.mapped 
# dev 
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/dev/dev.mt.bpe < $QE_DATA/dev/dev.tags > $QE_DATA/dev/dev.tags.mapped 
# test
python scripts/split_labels_by_subword_segmentation.py -t $QE_DATA/test/test.mt.bpe < $QE_DATA/test/test_words.tags > $QE_DATA/test/test.tags.mapped 
```
#### Recovering the original tag schema

Reduce fine-grained tags to {OK,BAD}
Evaluate using WMT F1 product evaluation script


