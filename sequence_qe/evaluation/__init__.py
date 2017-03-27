"""
Functions for doing evaluation on QE
"""

import re


def reduce_to_binary_labels(label_seqs):
    reduced_labels = [[re.sub(r'.*(OK|BAD)$', r'\1', label) for label in seq] for seq in label_seqs]
    return reduced_labels


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
            if pred == u'<UNK>':
                output_seq.append(u'{}-BAD'.format(tok))
            elif tok == pred:
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

    unsegmented_labels = []
    unsegmented_mt = []
    for label_seq, token_words in zip(labels, mt):
        mapped_labels = []
        mapped_mt = []

        current_labels = []
        current_word = []
        for label, word in zip(label_seq, token_words):
            if word.endswith(segmentation_suffix):
                current_labels.append(label)
                current_word.append(word)
            else:
                if len(current_labels) > 0:
                    current_labels.append(label)
                    current_word.append(word)
                    # word is finished, now use heuristic to decide which label wins
                    if heuristic == 'any_bad':
                        if any(l == u'BAD' for l in current_labels):
                            mapped_labels.append(u'BAD')
                        else:
                            mapped_labels.append(u'OK')
                    else:
                        raise ValueError('Unknown heuristic: {}'.format(heuristic))

                    mapped_mt.append(u''.join([re.sub(segmentation_suffix, '', u) for u in current_word]))
                    current_labels = []
                    current_word = []
                else:
                    mapped_labels.append(label)
                    mapped_mt.append(word)

        unsegmented_labels.append(mapped_labels)
        unsegmented_mt.append(mapped_mt)

    return unsegmented_labels, unsegmented_mt





