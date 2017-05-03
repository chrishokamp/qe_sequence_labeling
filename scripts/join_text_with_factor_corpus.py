"""
Join the tokens in parallel files (text, factor_corpus)

TODO: Where the factor delimiter exists inside a token or factor, try to map it to something else(?)

"""

import codecs
import logging
import sys
import argparse

from sequence_qe.dataset import parallel_iterator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run(text_file, factors_file, output_suffix, factor_delimiter=u'|'):
    text = codecs.open(text_file, encoding='utf8')
    factors = codecs.open(factors_file, encoding='utf8')
    output = codecs.open(text_file + '.{}'.format(output_suffix), 'w', encoding='utf-8')

    for text_tokens, factor_tokens in parallel_iterator(text, factors):
        assert len(text_tokens) == len(factor_tokens), 'Text and factor tokens must match'
        output.write(u' '.join([u'{}{}{}'.format(t_tok, factor_delimiter, f_tok)
                               for t_tok, f_tok in zip(text_tokens, factor_tokens)]) + u'\n')
    logger.info('Wrote new factor corpus to: {}'.format(output.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", help="The segmented text")
    parser.add_argument("-f", "--factors", help="The unsegmented factor corpus")
    parser.add_argument("--suffix", default="factor_corpus", required=False, help="A suffix for the mapped factor corpus output")

    args = parser.parse_args()
    run(args.text, args.factors, args.suffix)
