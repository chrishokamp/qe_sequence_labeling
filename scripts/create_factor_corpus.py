"""
Given two sentence and token-aligned parallel corpora, output a corpus where each token is factor1|factor2

"""

import codecs
import logging
import sys
import argparse
import json

import numpy as np


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from sequence_qe.dataset import parallel_iterator


def concat_tokens(seq1, seq2):
    assert len(seq1) == len(seq2)
    return [u'|'.join([tok1, tok2]) for tok1, tok2 in zip(seq1, seq2)]


# WORKING: remember to tie mt + output embeddings, and to try to initialize embeddings from pre-trained models
def main(factor_1_file, factor_2_file, factor_corpus_output):

    factor_1 = codecs.open(factor_1_file, encoding='utf8')
    factor_2 = codecs.open(factor_2_file, encoding='utf8')
    factor_iter = parallel_iterator(factor_1, factor_2)

    output = codecs.open(factor_corpus_output, 'w', encoding='utf8')

    count = 0
    for factor_1_seq, factor_2_seq in factor_iter:
        output.write(u' '.join(concat_tokens(factor_1_seq, factor_2_seq)) + '\n')

        count += 1
        if count % 10000 == 0:
            logger.info('Processed {} parallel rows'.format(count))

    output.close()
    logger.info('Wrote factored corpus to: {}'.format(factor_corpus_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", help="The first input factor")
    parser.add_argument("--f2", help="The second input factor")
    parser.add_argument("--output", help="Where the new corpus should written")

    args = parser.parse_args()
    main(args.f1, args.f2, args.output)
