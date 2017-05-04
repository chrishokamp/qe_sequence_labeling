'''
Parse the concatenated src+mt data with factors

Note that

'''


import os
import codecs
import logging
import argparse
import re

import spacy
from sequence_qe.dataset import mkdir_p

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_factors(line, nlp, factor_separator=u'|', factor_separator_replacement='-BAR-'):
    doc = nlp(line.strip())
    factors = [(w.orth_, w.tag_, w.dep_, w.head.tag_) for w in doc]
    # if `factor_separator` occurs in a token, map it to factor_separator_replacement
    factors = [tuple([re.sub(re.escape(factor_separator), factor_separator_replacement, f)
                      for f in w_factors])
               for w_factors in factors]
    return factors


def generate_spacy_factor_corpus(text_file, output_dir, lang_code, prefix, factor_separator=u'|'):
    mkdir_p(output_dir)
    text_output = codecs.open(os.path.join(output_dir, prefix + '.{}.'.format(lang_code) + 'tok'), 'w', encoding='utf8')
    factor_output = codecs.open(os.path.join(output_dir, prefix + '.{}.'.format(lang_code) + 'factors'), 'w', encoding='utf8')

    nlp = spacy.load(lang_code)
    logger.info('Loaded Spacy {} model'.format(lang_code))

    with codecs.open(text_file, encoding='utf8') as inp:

        for count, line in enumerate(inp):
            row = extract_factors(line, nlp)
            text, factors = zip(*[(factor_tup[0], factor_tup[1:]) for factor_tup in row])
            text_output.write(u' '.join(text) + '\n')
            factor_output.write(u' '.join([factor_separator.join(f) for f in factors]) + '\n')
            if (count + 1) % 1000 == 0:
                logger.info('Processed {} rows'.format(count + 1))
            
    logger.info('Wrote new files: {} and {}'.format(text_output.name, factor_output.name))
    text_output.close()
    factor_output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file with raw text")
    parser.add_argument("-o", "--output", help="Directory where output files will be written")
    parser.add_argument("-l", "--lang", help="Two-character language code used to load the correct Spacy model")
    parser.add_argument("-p", "--prefix", help="prefix for output files")

    args = parser.parse_args()
    generate_spacy_factor_corpus(args.input, args.output, args.lang, args.prefix)
