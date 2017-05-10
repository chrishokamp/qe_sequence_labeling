import numpy
import os
import sys

from nematus.nmt import train
from sequence_qe.dataset import mkdir_p

# the size of the src+mt concatenated data vocab
SRC_AND_MT_VOCAB_SIZE = 38559
FACTOR_1_SIZE = 308 
FACTOR_2_SIZE = 508
FACTOR_3_SIZE = 308

PE_VOCAB_SIZE = 41101
# never early stop
PATIENCE = 99999999999

INPUT_VOCAB_DIR = "/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus"
OUTPUT_VOCAB_DIR = "/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models"
SRC_VOCAB = os.path.join(INPUT_VOCAB_DIR, 'train.src-mt.concatenated.bpe.json')
FACTOR_1_VOCAB = os.path.join(INPUT_VOCAB_DIR, 'factor_1.json')
FACTOR_2_VOCAB = os.path.join(INPUT_VOCAB_DIR, 'factor_2.json')
FACTOR_3_VOCAB = os.path.join(INPUT_VOCAB_DIR, 'factor_3.json')
PE_VOCAB = os.path.join(OUTPUT_VOCAB_DIR, 'mt-pe/vocab.pe.json')

TRAIN_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/4M/spacy_factor_corpus"
SRC_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.src-mt.concatenated.bpe.factor_corpus')
TRG_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.pe.prepped.orig')

# WMT 16 EN-DE QE/APE DEV Data
QE_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017/spacy_factor_corpus"
SRC_DEV = os.path.join(QE_DATA_DIR, 'dev.src-mt.concatenated.bpe.factor_corpus')
TRG_DEV = os.path.join(QE_DATA_DIR, 'dev.pe.prepped.orig')

if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=512,
                    dim=1028,
                    n_words=PE_VOCAB_SIZE,
                    n_words_src=SRC_AND_MT_VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=100,
                    batch_size=32,
                    valid_batch_size=32,
                    datasets=[SRC_TRAIN, TRG_TRAIN],
                    valid_datasets=[SRC_DEV, TRG_DEV],
                    dictionaries=[SRC_VOCAB, FACTOR_1_VOCAB, FACTOR_2_VOCAB, FACTOR_3_VOCAB, PE_VOCAB],
                    factors=4,
                    dim_per_factor=[256, 64, 128, 64],
                    validFreq=5000,
                    dispFreq=500,
                    saveFreq=5000,
                    sampleFreq=5000,
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
