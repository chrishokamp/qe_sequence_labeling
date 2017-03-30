import numpy
import os
import sys

from nematus.nmt import train

VOCAB_SIZE = 32000
SRC = "en"
TGT = "de"
DATA_DIR = "/media/1tb_drive/parallel_data/en-de/google_seq2seq_dataset"

SRC_VOCAB = os.path.join(DATA_DIR, 'train.tok.clean.bpe.32000.en.json')
TRG_VOCAB = os.path.join(DATA_DIR, 'train.tok.clean.bpe.32000.de.json')

SRC_TRAIN = os.path.join(DATA_DIR, 'train.tok.clean.bpe.32000.en')
TRG_TRAIN = os.path.join(DATA_DIR, 'train.tok.clean.bpe.32000.de')

SRC_DEV = os.path.join(DATA_DIR, 'newstest2015.tok.clean.bpe.32000.en')
TRG_DEV = os.path.join(DATA_DIR, 'newstest2015.tok.clean.bpe.32000.de')


if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=256,
                    dim=1024,
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=64,
                    valid_batch_size=64,
                    datasets=[SRC_TRAIN, TRG_TRAIN],
                    valid_datasets=[SRC_DEV, TRG_DEV],
                    dictionaries=[SRC_VOCAB, TRG_VOCAB],
                    validFreq=10000,
                    dispFreq=100,
                    saveFreq=30000,
                    sampleFreq=10000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
