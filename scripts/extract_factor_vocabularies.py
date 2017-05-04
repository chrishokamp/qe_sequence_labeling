"""
Parse the concatenated src+mt factored corpus data

Split out factors by the `factor_separator`, and get all tokens in the factor
Add UNK and BOS/EOS tokens to match nematus behavior
"""


import logging
import argparse
import json

import numpy

from collections import OrderedDict

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def factor_iter(factor_file, num_factors, factor_separator=u'|'):
    for row in factor_file:
        parsed_row = [factor_tup.split(factor_separator) for factor_tup in row.split()]
        assert all(len(f) == num_factors for f in parsed_row)
        yield parsed_row


def vocab_dictionaries_from_factor_iterator(factor_iterator, num_factors):
    token_freqs = [OrderedDict() for _ in range(num_factors)]

    for factor_row in factor_iterator:
        for factor_tup in factor_row:
            for idx, factor in enumerate(factor_tup):
                if factor not in token_freqs[idx]:
                    token_freqs[idx][factor] = 0
                else:
                    token_freqs[idx][factor] += 1

    token_dicts = [OrderedDict() for _ in range(num_factors)]

    for i, freq_dict in enumerate(token_freqs):
        tokens = freq_dict.keys()
        freqs = freq_dict.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [tokens[ii] for ii in sorted_idx[::-1]]

        token_dicts[i]['eos'] = 0
        token_dicts[i]['UNK'] = 1
        for ii, ww in enumerate(sorted_words):
            token_dicts[i][ww] = ii+2

    return token_dicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file with factor tuples separated by `factor_separator`")
    parser.add_argument("-o", "--output", help="Directory where output files will be written")
    parser.add_argument("-n", "--num_factors", help="the number of factors")

    args = parser.parse_args()

    factor_iterator = factor_iter(open(args.input, 'r'), args.num_factors)
    factor_dicts = vocab_dictionaries_from_factor_iterator(factor_iterator, num_factors=args.num_factors)

    for idx, filename in enumerate(['factor_{}'.format(i+1) for i in range(args.num_factors)]):
        with open('%s.json'%filename, 'wb') as f:
            json.dump(factor_dicts[idx], f, indent=2, ensure_ascii=False)
        logger.info('Wrote index to: {}'.format(filename))
