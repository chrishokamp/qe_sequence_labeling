import numpy
import os
import sys

from nematus.nmt import train
from sequence_qe.dataset import mkdir_p

# are src and trg vocab sizes actually required? -- these could be obtained directly from vocab indices
# For AmuNMT WMT 2016 pretrained models
SRC_VOCAB_SIZE=40587
MT_VOCAB_SIZE=40822
PE_VOCAB_SIZE=41101

PATIENCE = 99999999999 #never early stop
SRC = "en"
TGT = "de"

VOCAB_DIR = "/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models"
SRC_VOCAB = os.path.join(VOCAB_DIR, 'src-pe/vocab.src.json')
MT_VOCAB = os.path.join(VOCAB_DIR, 'mt-pe/vocab.mt.json')
PE_VOCAB = os.path.join(VOCAB_DIR, 'mt-pe/vocab.pe.json')

TRAIN_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/amunmt_artificial_ape_2016/data/500K_and_20x_task_internal"
SRC_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.mt.factor_corpus')
TRG_TRAIN = os.path.join(TRAIN_DATA_DIR, 'train.pe.prepped')

# WMT 16 EN-DE QE/APE DEV Data
QE_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017"
SRC_DEV = os.path.join(QE_DATA_DIR, 'dev.mt.factor_corpus')
TRG_DEV = os.path.join(QE_DATA_DIR, 'dev.pe.prepped')

mkdir_p('model')

# start training from best model from previous experiment
STARTING_MODEL = '/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de/model/model.npz.npz.best_bleu'

if __name__ == '__main__':
    validerr = train(saveto=os.path.join('model/model.npz'),
                     prior_model=STARTING_MODEL,
                     reload_=True,
                     dim_word=256,
                     dim=512,
                     n_words=PE_VOCAB_SIZE,
                     n_words_src=SRC_VOCAB_SIZE,
                     decay_c=0.,
                     clip_c=1.,
                     lrate=0.0001,
                     optimizer='adadelta',
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     datasets=[SRC_TRAIN, TRG_TRAIN],
                     valid_datasets=[SRC_DEV, TRG_DEV],
                     dictionaries=[MT_VOCAB, SRC_VOCAB, PE_VOCAB],
                     tie_encoder_decoder_embeddings=False,
                     factors=2,
                     dim_per_factor=[128, 128],
                     validFreq=1000,
                     dispFreq=500,
                     saveFreq=5000,
                     sampleFreq=1000,
                     use_dropout=False,
                     dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                     dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                     dropout_source=0.1, # dropout source words (0: no dropout)
                     dropout_target=0.1, # dropout target words (0: no dropout)
                     overwrite=False,
                     external_validation_script='./fine_tune_validate.sh')
    print validerr
