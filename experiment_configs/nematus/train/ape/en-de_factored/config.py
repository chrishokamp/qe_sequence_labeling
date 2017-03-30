import numpy
import os
import sys

from nematus.nmt import train

VOCAB_SIZE = 90000
SRC = "en"
TGT = "de"

VOCAB_DIR = "/media/1tb_drive/nematus_ape_experiments/pretrained_wmt16_models/en-de"
SRC_VOCAB = os.path.join(VOCAB_DIR, 'vocab.en.json')
TRG_VOCAB = os.path.join(VOCAB_DIR, 'vocab.de.json')

TRAIN_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/concat_500k_with_wmt16/factored_ape_corpus"
SRC_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.mt_aligned_with_source.factor')
TRG_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.pe.prepped')

# WMT 16 EN-DE QE Data
QE_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/wmt_2016/dev_wmt16_pretrained_bpe"
SRC_DEV = os.path.join(QE_DATA_DIR, 'dev.mt_aligned_with_source.factor')
TRG_DEV = os.path.join(QE_DATA_DIR, 'dev.pe.prepped')

if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=512,
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
                    dictionaries=[TRG_VOCAB, SRC_VOCAB, TRG_VOCAB],
                    tie_encoder_decoder_embeddings=False,
                    factors=2,
                    dim_per_factor=[256,256],
                    validFreq=500,
                    dispFreq=100,
                    saveFreq=5000,
                    sampleFreq=1000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
