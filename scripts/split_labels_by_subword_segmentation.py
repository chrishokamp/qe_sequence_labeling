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



def map_tags(segmented_target, original_tags, segmentation_suffix=u'@@'):
    """
    Maps tags from the original segmentation into a new tag sequence, performing two operations:
    (1) append the corresponding target word to the tag
    (2) if the original corresponding word was segmented, duplicate the tag

    Params:
      segmented_target: a string where some tokens may end with `segmentation_suffix`
      original_tags: the tags corresponding to the target sequence before segmentation
      segmentation_suffix: the suffix used to indicate that a tag was segmented

    Returns:
      the mapped tag sequence
    """

    # assume whitespace tokenization
    segmented_target = segmented_target.split()
    original_tags = original_tags.split()

    mapped_tags = []
    tag_idx = 0
    for trg_word in segmented_target:
        mapped_tag = u'{}-{}'.format(trg_word, original_tags[tag_idx])
        mapped_tags.append(mapped_tag)
        if not trg_word.endswith(segmentation_suffix):
            tag_idx += 1

    assert tag_idx == len(original_tags), 'We must cover all of the original tags'
    assert len(mapped_tags) == len(segmented_target), 'We must have a tag for every word'
    return mapped_tags


def map_dataset(target_file, tags_file):
    data_iter = parallel_iterator(target_file, tags_file)
    for target, tags in data_iter:
        yield map_tags(target, tags)


def run(target_file, tags_file, output):
    target_file = codecs.open(target_file, encoding='utf8')
    if tags_file.name != '<stdin>':
        tags_file = codecs.open(tags_file.name, encoding='utf8')
    if output.name != '<stdout>':
        output = codecs.open(output.name, 'w', encoding='utf-8')

    for tag_seq in map_dataset(target_file, tags_file):
        output.write((u' '.join(tag_seq) + u'\n').encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="The segmented target text")
    parser.add_argument(
        '--labels', '-l', type=argparse.FileType('r'), default=sys.stdin,
        help="Input containing one label per word, corresponding to the target words _before_ segmentation " +
             "(default: standard input).")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout,
                        help="Where we should write the new mapped tags")

    args = parser.parse_args()
    run(args.target, args.labels, args.output)
