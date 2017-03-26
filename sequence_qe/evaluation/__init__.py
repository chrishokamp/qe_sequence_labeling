"""
Functions for doing evaluation on QE
"""

import re


def reduce_to_binary_labels(label_seqs):
    reduced_labels = [[re.sub(r'.*(OK|BAD)$', r'\1', label) for label in seq] for seq in label_seqs]
    return reduced_labels


# WORKING: filter functions which can be applied as optional post-processing on a qe data stream
def non_matching_words_are_bad(mt, label_seqs):
    """
    When the token portion of the label doesn't match the MT token, always label as 'BAD'
    """

    # extract the word portion of the labels
    label_words = [[re.sub(r'(.*)-(OK|BAD)$', r'\1', label) for label in seq] for seq in label_seqs]
    output_seqs = []
    for token_words, pred_words, label_seq in zip(mt, label_words, label_seqs):
        output_seq = []
        for tok, pred, label in zip(token_words, pred_words, label_seq):
            if tok == pred:
                output_seq.append(label)
            else:
                output_seq.append(u'{}-BAD'.format(tok))
        output_seqs.append(output_seq)

    return output_seqs


def unsegment_labels(labels, mt, segmentation_suffix=u'@@', heuristic='any_bad'):
    """
    Reduce label sequence according to MT segmentation, apply heuristic to decide how labels should be reduced

    Params:
      labels: sequence of labels
      mt: sequence of tokens with segmentation applied
      segmentation_suffix: suffix which indicates when a token has been segmented
      heuristic: key which tells us how to reduce segmented labels

    Returns:
      reduced labels corresponding to the original MT hypothesis without segmentation
    """

    # for token_words, pred_words, label_seq in zip(mt, label_words, label_seqs):
    pass

