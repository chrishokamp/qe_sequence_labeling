import sys
import codecs
import os
import subprocess
import argparse
import pdb

# Path to TERCOM.
#PATH_TERCOM = '~/workspace/tercom-0.7.25/tercom.7.25.jar'
PATH_TERCOM = 'tercom-0.7.25/tercom.7.25.jar'

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(prog='TERCOM wrapper',
                       description='A Python wrapper for TERCOM. Uses edit '
                       'distance to align a target sentence to a post-edited '
                       'sentence.')
    parser.add_argument('-target_file', type=str, required=True)
    parser.add_argument('-post_edited_file', type=str, required=True)
    args = vars(parser.parse_args())

    target_file = args['target_file']
    post_edited_file = args['post_edited_file']

    tokenized_target_sentences = []
    tokenized_post_edited_sentences = []

    # Read original data, one sentence per line.
    with codecs.open(target_file, 'r', 'utf8') as f_t, \
         codecs.open(post_edited_file, 'r', 'utf8') as f_pe:
        for line_t, line_pe in zip(f_t, f_pe):
            t = line_t.rstrip('\n')
            pe = line_pe.rstrip('\n')
            tokenized_target_sentences.append(t)
            tokenized_post_edited_sentences.append(pe)

    # Create hypothesis and reference files for TERCOM.
    hypothesis_file = '%s.tercom.hyp' % target_file
    reference_file = '%s.tercom.ref' % post_edited_file
    with codecs.open(hypothesis_file, 'w', 'utf8') as f_hyp, \
         codecs.open(reference_file, 'w', 'utf8') as f_ref:
        for i, (hyp, ref) in enumerate(zip(tokenized_target_sentences,
                                           tokenized_post_edited_sentences)):
            f_hyp.write('%s\t(%.12d)\n' % (hyp, i))
            f_ref.write('%s\t(%.12d)\n' % (ref, i))

    # Run TERCOM.
    output_prefix = '%s.tercom.out' % post_edited_file
    cmd = 'java -jar %s -r %s -h %s -n %s -d 0' % (PATH_TERCOM,
                                                   reference_file,
                                                   hypothesis_file,
                                                   output_prefix)
    p = subprocess.Popen(cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()

