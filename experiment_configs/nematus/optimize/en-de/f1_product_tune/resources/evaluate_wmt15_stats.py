from __future__ import division, print_function

import os
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#------------------ evaluation for the WMT15 format:------------------------
#
#       <METHOD NAME> <SEGMENT NUMBER> <WORD INDEX> <WORD> <BINARY SCORE>
#       tab-separated, no empty lines
#
#---------------------------------------------------------------------------


#-------------PREPROCESSING----------------
# check if <a_list> is a list of lists
def list_of_lists(a_list):
    if isinstance(a_list, (list, tuple, np.ndarray)) and len(a_list) > 0 and all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False


# check that two lists of sequences have the same number of elements
def check_word_tag(words_seq, tags_seq, dataset_name=''):
    assert(len(words_seq) == len(tags_seq)), "Number of word and tag sequences doesn't match in %s" % dataset_name
    for idx, (words, tags) in enumerate(zip(words_seq, tags_seq)):
        assert(len(words) == len(tags)), "Numbers of words and tags don't match in sequence %d of %s" % (idx, dataset_name)


# check that words in reference and prediction match
def check_words(ref_words, pred_words):
    assert(len(ref_words) == len(pred_words)), "Number of word sequences doesn't match in reference and hypothesis: %d and %d" % (len(ref_words), len(pred_words))
    for idx, (ref, pred) in enumerate(zip(ref_words, pred_words)):
        ref_str = ' '.join(ref).lower()
        pred_str = ' '.join(pred).lower()
        assert(ref_str == pred_str), "Word sequences don't match in reference and hypothesis at line %d:\n\t%s\n\t%s\n" % (idx, ref_str, pred_str)


def parse_submission(ref_txt_file, ref_tags_file, submission):
    tag_map = {'OK': 1, 'BAD': 0}
    # parse txt
    true_words = []
    for line in open(ref_txt_file):
        line = line.decode('utf-8').strip('\n')
        # if target txt with phrase segmentation provided
        if line.find(' || ') != -1:
            phr = line.split(' || ')
            true_words.append(flatten([p.split() for p in phr]))
        # plaintext target
        else:
            true_words.append(line.split())

    # parse test tags
    true_tags = []
    for line in open(ref_tags_file):
        true_tags.append([tag_map[t] for t in line[:-1].decode('utf-8').split()])
    check_word_tag(true_words, true_tags, dataset_name='reference')

    # parse and check the submission
    phrase = False
    test_tags = [[] for i in range(len(true_tags))]
    test_words = [[] for i in range(len(true_tags))]
    for idx, line in enumerate(open(submission)):
        chunks = line[:-1].decode('utf-8').strip('\r').split('\t')
        cur_seq = int(chunks[1])
        words = chunks[3].strip().split()
        if len(words) > 1:
            phrase = True
        test_tags[cur_seq].extend([tag_map[chunks[4]] for i in range(len(words))])
        test_words[cur_seq].extend(words)
    check_words(true_words, test_words)
    check_word_tag(test_words, test_tags, dataset_name='hypothesis')
    if phrase:
        print("Phrase-level labels have been detected")
    return true_tags, test_tags


#---------------------------EVALUATION-------------------------
# convert list of lists into a flat list
def flatten(lofl):
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags):
    for true, test in zip(true_tags, test_tags):
        scores = flatten(confusion_matrix(true, test))
        if len(scores) == 1:
            print("0 0 0", scores[0])
        else:
            print("%d %d %d %d" % tuple(scores))
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    #print("F1-score multiplied: ", f1_bad * f1_good)
    #print("F1-BAD: ", f1_bad)


def evaluate(ref_txt_file, ref_tags_file, submission):
    true_tags, test_tags = parse_submission(ref_txt_file, ref_tags_file, submission)
    compute_scores(true_tags, test_tags)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ref_txt", action="store", help="test target text (one line per sentence)")
    parser.add_argument("ref_tags", action="store", help="test WORD-LEVEL labels (one line per sentence)")
    parser.add_argument("submission", action="store", help="submission (wmt15 format)")
    args = parser.parse_args()

    #print("Evaluating '%s'" % os.path.basename(args.submission))
    evaluate(args.ref_txt, args.ref_tags, args.submission)
