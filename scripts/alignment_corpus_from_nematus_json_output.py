"""
Given a Nematus JSON file, output the top aligned target token for each source token.

Hypothesis:
We preserve source order, because the output is intended to be used as a factor for an NMT
system, and we hypothesize that the source order is more important than the target order for Automatic Post-Editing

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


# this is here to fix the broken Nematus JSON
def fix_JSON(json_message=None, unk_token='UNK'):
    result = None
    try:
        result = json.loads(json_message)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(e.message.split(' ')[-1].replace(')',''))
        # Remove the offending character:
        json_message = list(json_message)
        json_message[idx_to_replace] = ' {} '.format(unk_token)
        new_message = ''.join(json_message)
        return fix_JSON(json_message=new_message)
    return result


def parse_json_row_file(json_row_file):
    """
    Parse a file where each row is a json string

    Params:
      json_row_file

    Returns:
      generator of file rows
    """
    with codecs.open(json_row_file, encoding='utf8') as json_file:
        for row in json_file:
            try:
                row = fix_JSON(json_message=row)
                yield row
            except:
                logger.error('Error parsing row')
                #print(row)
                yield None


def sort_alignment_weights(alignment_weights):
    """"
    Sort a matrix of alignment weights at each timestep. The output
    is a 2d array, where each row is the list of token _indexes_ for that timestep sorted in descending
    order of their alignment weights.

    Params:
      alignment_weights: a 2d numpy array of weights

    Returns:
      2d array where each row is the decending argsort of that timestep

    """

    sorted_weights = np.zeros(alignment_weights.shape, dtype='int16')
    for i, row in enumerate(alignment_weights):
        sorted_indices = np.argsort(row)[::-1]
        sorted_weights[i] = sorted_indices
    return sorted_weights


def extract_word_alignment(source_sequence, target_sequence, alignments, order='source'):
    """
    Extract the word alignment at each timestep, according to alignment weights.

    Params:
      source_sequence: sequence of source tokens
      target_sequence: sequence of target tokens -- one of (source_sequence, target_sequence) will not be used
      alignments: a 2d matrix of alignment weights, with dimensionality (target_len, source_len)
      order: the output order -- if 'source', output one target token for each source timestep.
        if 'target', output one source token for each target timestep.

    Returns:
      sequence of tokens with length = len(source_sequence) or len(target_sequence)
    """
    if order not in set(['source', 'target']):
        raise ValueError

    if order == 'source':
        ref_seq = source_sequence
        align_seq = target_sequence
        sorted_weights = sort_alignment_weights(alignments.T)
    else:
        ref_seq = target_sequence
        align_seq = source_sequence
        sorted_weights = sort_alignment_weights(alignments)

    assert len(sorted_weights) == len(ref_seq)
    assert len(sorted_weights[0]) == len(align_seq)

    aligned_tokens = []
    for row in sorted_weights:
        aligned_tokens.append(align_seq[row[0]])

    return aligned_tokens


def main(nematus_json_file, output_file, order='source', eos_token=u'eos'):

    output = codecs.open(output_file, 'w', encoding='utf8')
    # Note: it's important that we start the row count at 1, because we'll use the row numbers afterward
    # to delete rows that had json parsing errors
    row_count = 1
    deleted_rows = open(nematus_json_file + '.deleted_rows', 'w')
    for row_obj in parse_json_row_file(nematus_json_file):
        if row_obj is None:
            logger.error('DELETED ROW: {}'.format(row_count))
            deleted_rows.write('{}\n'.format(row_count))
            row_count += 1
            continue
        source_tokens = row_obj['source_sent'].split() + [eos_token]
        target_tokens = row_obj['target_sent'].split() + [eos_token]
        alignment_weights = np.array(row_obj['matrix'])

        aligned_tokens = extract_word_alignment(source_tokens, target_tokens, alignment_weights, order=order)
        # cutoff the eos token again, so that we have one-to-one correspondence with the mt input data
        aligned_tokens = aligned_tokens[:-1]
        output.write(u' '.join(aligned_tokens) + '\n')
        row_count += 1
        if row_count % 10000 == 0:
            logger.info('Processed {} rows'.format(row_count))


    output.close()
    deleted_rows.close()
    logger.info('Wrote aligned tokens in {} order to: {}'.format(order, output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="A nematus json file with fields for source, target, and alignment weights")
    parser.add_argument("--output", help="Where to write the output")
    parser.add_argument("--order", help="One of {source|target} -- the order of the alignment output")

    args = parser.parse_args()
    main(args.json, args.output, order=args.order)
