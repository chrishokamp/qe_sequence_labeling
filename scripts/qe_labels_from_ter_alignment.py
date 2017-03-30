"""
Call TER to generate QE labels given parallel files of mt hyps and post-edited references
"""

import subprocess
import sys
import logging
import argparse
import os
import codecs
import cgi

from sequence_qe.dataset import parse_pra_xml, mkdir_p, parallel_iterator


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_ter_alignment(hyps_file, refs_file, output_path, src_lang, trg_lang, tercom_path):

    tercom_jar = os.path.join(tercom_path, 'tercom.7.25.jar')

    mkdir_p(output_path)
    output_prefix = os.path.join(output_path, '{}-{}.tercom.out'.format(src_lang, trg_lang))

    # WORKING: we need to put hyps and refs files in a special format
    hyps_file_iter = codecs.open(hyps_file, encoding='utf8')
    refs_file_iter = codecs.open(refs_file, encoding='utf8')
    hyp_ref_iter = parallel_iterator(hyps_file_iter, refs_file_iter)

    temp_hyps_file = hyps_file + '.ter.temp'
    temp_refs_file = refs_file + '.ter.temp'
    with codecs.open(temp_hyps_file, 'w', encoding='utf8') as f_hyp:
        with codecs.open(temp_refs_file, 'w', encoding='utf8') as f_ref:
            for i, (hyp, ref) in enumerate(hyp_ref_iter):
                # Note the logic for escaping XML entities here
                f_hyp.write('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in hyp]), i))
                f_ref.write('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in ref]), i))

    # Run TERCOM.
    cmd = 'java -jar {} -r {} -h {} -n {} -d 0'.format(tercom_jar,
                                                       temp_refs_file,
                                                       temp_hyps_file,
                                                       output_prefix)
    p = subprocess.Popen(cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()

    os.remove(temp_hyps_file)
    os.remove(temp_refs_file)

    # Parse TERCOM output xml
    mt_tokens, pe_tokens, edits, hters = \
        parse_pra_xml.parse_file('{}.xml'.format(output_prefix))

    # WORKING HERE
    tags_map = {'C': 'OK', 'S': 'BAD', 'I': 'BAD', 'D': 'BAD'}
    tags = [parse_pra_xml.get_tags(edit, tags_map, keep_inserts=False) for edit in edits]

    tags_output_file = os.path.join(output_path, output_prefix + '.tags')
    with codecs.open(tags_output_file, 'w', encoding='utf8') as out:
        for row in tags:
            out.write(u' '.join(row) + u'\n')
    logger.info('Wrote tags to: {}'.format(tags_output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps", help="The target language code")
    parser.add_argument("--refs", help="The target language code")
    parser.add_argument("--output", help="Where to output the TER alignment files")
    parser.add_argument("--prefix", help="The prefix to use for the output data")
    parser.add_argument("--src_lang", help="The source language code")
    parser.add_argument("--trg_lang", help="The target language code")
    parser.add_argument("--tercom", help="The path to the directory containing tercom.7.25.jar")

    args = parser.parse_args()
    extract_ter_alignment(args.hyps, args.refs, args.output, args.src_lang, args.trg_lang, args.tercom)

