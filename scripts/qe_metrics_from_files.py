"""
Given a file containing hyps and a file containing gold tags, output QE metrics to file

"""

import codecs
import logging
import argparse
import json

from sequence_qe.evaluation import accuracy, f1_scores
from sequence_qe.dataset import tokens_from_file


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(hyps_file, gold_file, output_prefix):
    hyps = tokens_from_file(hyps_file)
    gold = tokens_from_file(gold_file)

    acc = accuracy(hyps, gold)

    class_f1s = f1_scores(hyps, gold)
    f1_bad = class_f1s['BAD']
    f1_ok = class_f1s['OK']

    f1_product = f1_bad * f1_ok

    evaluation_record = {
        'accuracy': acc,
        'f1_bad': f1_bad,
        'f1_ok': f1_ok,
        'f1_product': f1_product
    }

    with codecs.open(output_prefix + '.json', 'w', encoding='utf8') as out:
        out.write(json.dumps(evaluation_record))

    print('ACC: {} F1_BAD: {} F1_OK: {} F1_PRODUCT: {}'.format(acc, f1_bad, f1_ok, f1_product))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps", help="file containing hypotheses tags")
    parser.add_argument("--gold", help="file containing gold tag")
    parser.add_argument("--output", help="prefix, output will be written to `prefix.json`")

    args = parser.parse_args()
    main(args.hyps, args.gold, args.output)
