"""
Functions for doing evaluation on QE
"""

import re

from sklearn.metrics import f1_score


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


def f1_scores(pred, true):
    """
    Compute per-class f1 scores

    Params:
      pred: 2d sequence of predictions
      true: 2d ground truth sequence

    Returns:
      {'OK': <f1_score of OK class>, 'BAD': <f1_score of BAD class>}
    """
    tag_map = {u'OK': 0, u'BAD': 1, u'<UNK>': 1}
    # assume 2d
    flat_preds = [tag_map[w] for s in pred for w in s]
    flat_true = [tag_map[w] for s in true for w in s]

    f1_good, f1_bad = f1_score(flat_true, flat_preds, average=None)

    # map classes to tags
    return {u'OK': f1_good, u'BAD': f1_bad}


def truncate_at_eos(mt, preds, true, eos_token=u'</S>'):
    """
    Cut sequences based on location of the eos token

    Params:
      mt: 2d sequence of mt tokens
      preds: 2d sequence of predicted
      true: 2d sequence of ground truth tokens
      eos_token: the symbol used as the EOS token

    Returns:
      (mt, preds, true) where each row is truncated according to the eos_token's position in `mt`
    """
    no_eos_mt = []
    no_eos_preds = []
    no_eos_true = []

    for m, p, t in zip(mt, preds, true):
        if eos_token in m:
            eos_idx = m.index(eos_token)
            m = m[:eos_idx]
            p = p[:eos_idx]
            t = t[:eos_idx]

        no_eos_mt.append(m)
        no_eos_preds.append(p)
        no_eos_true.append(t)

    return no_eos_mt, no_eos_preds, no_eos_true


def qe_output_evaluation(mt, preds, true):
    """
    Return a score report for the output of a QE Model

    :param mt:
    :param pred:
    :param true:
    :return:
    """

    # map predictions where the token portion doesn't match the mt to 'BAD'
    mapped_preds = non_matching_words_are_bad(mt, preds)

    # reduce expanded tagset to {OK, BAD}
    reduced_preds = reduce_to_binary_labels(mapped_preds)
    reduced_true = reduce_to_binary_labels(true)

    # remove the model internal segmentation so that the data matches the orignal
    no_seg_preds, orig_mt = unsegment_labels(reduced_preds, mt)
    no_seg_true, orig_mt = unsegment_labels(reduced_true, mt)

    # get the f1 scores
    class_f1s = f1_scores(no_seg_preds, no_seg_true)
    f1_bad = class_f1s['BAD']
    f1_ok = class_f1s['OK']

    f1_product = f1_bad * f1_ok

    evaluation_record = {
        'f1_bad': f1_bad,
        'f1_ok': f1_ok,
        'f1_product': f1_product
    }

    return evaluation_record









