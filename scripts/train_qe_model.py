"""
Train a QE model with checkpointing

"""

import codecs
import logging
import os
import argparse

from sequence_qe.dataset import parallel_iterator

from sequence_qe.models import UnidirectionalAttentiveQEModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: load yaml config for experiment

def qe_data_iterator_func(source_file, target_file, labels_file):

    def _build_iterator():
        src_f = codecs.open(source_file, encoding='utf8')
        trg_f = codecs.open(target_file, encoding='utf8')
        labels_f = codecs.open(labels_file, encoding='utf8')
        return parallel_iterator(src_f, trg_f, labels_f)

    return _build_iterator


def get_training_files(train_dir):
    source = os.path.join(train_dir, 'train.src.bpe')
    mt = os.path.join(train_dir, 'train.mt.bpe')
    labels = os.path.join(train_dir, 'train.tags.mapped')
    return source, mt, labels


def get_dev_files(dev_dir):
    source = os.path.join(dev_dir, 'dev.src.bpe')
    mt = os.path.join(dev_dir, 'dev.mt.bpe')
    labels = os.path.join(dev_dir, 'dev.tags.mapped')
    return source, mt, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Directory containing training data")
    parser.add_argument("-v", "--validation", help="Directory containing validation data")
    parser.add_argument("-l", "--logdir", help="Directory for output data")
    parser.add_argument("-r", "--resources", help="Location of resources -- vocabulary indexes")
    args = parser.parse_args()

    train_files = get_training_files(args.train)
    dev_files = get_dev_files(args.validation)
    train_iter_func = qe_data_iterator_func(*train_files)
    dev_iter_func = qe_data_iterator_func(*dev_files)

    # TODO: supply external config for QE model
    config = {
        'resources': args.resources
    }

    model = UnidirectionalAttentiveQEModel(storage=args.logdir, config=config)

    # WORKING HERE: init QE model, train QE model
    model.train(train_iter_func=train_iter_func, dev_iter_func=dev_iter_func)

