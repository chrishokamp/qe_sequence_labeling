"""
Given an input vocabulary index, duplicate each entry, creating OK-<word> and BAD-<word> entries

The <S>, </S> and <UNK> tokens are the only ones exempt from this duplication, it's up to downstream applications to decide what to do with these

Input: vocab index in the target language

Output: new vocab index ~2x size of input index

"""

import logging
import sys
import argparse
import cPickle
from collections import OrderedDict

from sequence_qe.dataset import parallel_iterator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def map_vocab_index_to_qe_vocab(vocab_index):
    qe_vocab_index = {}
    special_tokens = set([u'<S>', u'<UNK>', u'</S>'])

    current_index = 0
    # if there is order in values of the original index, we try to preserve it
    for w, idx in sorted(vocab_index.items(), key=lambda x: x[1]):
        if w in special_tokens:
            qe_vocab_index[w] = current_index
            current_index += 1
        else:
            ok_tok = u'{}-OK'.format(w)
            bad_tok = u'{}-BAD'.format(w)
            qe_vocab_index[ok_tok] = current_index
            current_index += 1
            qe_vocab_index[bad_tok] = current_index
            current_index += 1

    assert len(qe_vocab_index) == (len(vocab_index)*2) - len(special_tokens)

    return qe_vocab_index


def run(input_index_filename, output_index_filename):
    original_index = cPickle.load(open(input_index_filename))
    qe_index = map_vocab_index_to_qe_vocab(original_index)

    with open(output_index_filename, 'w') as out:
        cPickle.dump(qe_index, out)

    logger.info('Wrote new index of size: {} to {}'.format(len(qe_index), output_index_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The original vocabulary index")
    parser.add_argument("-o", "--output", help="The location where the new QE vocab index should be saved")
    args = parser.parse_args()
    run(args.input, args.output)
