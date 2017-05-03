"""
Duplicate QE labels and map to extended QE vocabulary

Input: segmented_target_file, original_labels_file

Output: new_labels_file, where labels are copied where necessary to match the target segmentation

"""

import codecs
import logging
import sys
import argparse

from sequence_qe.dataset import parallel_iterator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def map_factors(segmented_text, original_factors, segmentation_suffix=u'@@', factor_separator=u'|'):
    """
    Maps factors from the original segmentation into a new tag sequence
    Where tags are split due to the segmentation, prepend B- and I- prefixes to indicate the 
      position of the factor in the segmentation

    Params:
      segmented_text: a string where some tokens may end with `segmentation_suffix`
      original_factors: factors joined by `factor_separator`, corresponding to the tokens before segmentation
      segmentation_suffix: the suffix used to indicate that a word was segmented
      factor_separator: the string used to separate factors in the factor corpus

    Returns:
      the mapped factor sequence
    """

    # assume whitespace tokenization
    if type(segmented_text) is not list:
        segmented_text = segmented_text.split()
    if type(original_factors) is not list:
        original_factors = original_factors.split()

    mapped_factors = []
    factor_idx = 0
    replicating = False
    for word in segmented_text:
        current_factor = original_factors[factor_idx]
        if word.endswith(segmentation_suffix):
            each_factor = current_factor.split(factor_separator)
            if replicating:
                new_factor = factor_separator.join([u'I-{}'.format(f) for f in each_factor])
            else:
                new_factor = factor_separator.join([u'B-{}'.format(f) for f in each_factor])
            mapped_factors.append(new_factor)
            replicating = True
        else:
            if replicating:
                each_factor = current_factor.split(factor_separator)
                new_factor = factor_separator.join([u'I-{}'.format(f) for f in each_factor])
                mapped_factors.append(new_factor)
                replicating = False
            else:
                mapped_factors.append(current_factor)
            factor_idx += 1
    # If we're still replicating, we need to increment the index for the assertions below
    if replicating:
        factor_idx += 1

    assert len(mapped_factors) == len(segmented_text), 'After mapping, we need one factor group per segmented token'
    assert factor_idx == len(original_factors), 'We must cover all of the original factors'

    return mapped_factors


# def map_factors(segmented_text, original_factors, segmentation_suffix=u'@@', factor_separator=u'|'):
def map_dataset(text, factors):
    data_iter = parallel_iterator(text, factors)
    for segmented_text, factor_text in data_iter:
        yield map_factors(segmented_text, factor_text)


def run(text_file, factors_file, output_suffix):
    text = codecs.open(text_file, encoding='utf8')
    factors = codecs.open(factors_file, encoding='utf8')
    output = codecs.open(factors_file + '.{}'.format(output_suffix), 'w', encoding='utf-8')

    for mapped_factors in map_dataset(text, factors):
        output.write(u' '.join(mapped_factors) + u'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", help="The segmented text")
    parser.add_argument("-f", "--factors", help="The unsegmented factor corpus")
    parser.add_argument("--suffix", default="bpe", required=False, help="A suffix for the mapped factor corpus output")

    args = parser.parse_args()
    run(args.text, args.factors, args.suffix)
