import os
import errno
import itertools
import codecs
import subprocess
import tempfile
import cgi
import sys

import parse_pra_xml


def parallel_iterator(*iters):
    for lines in itertools.izip(*iters):
        lines = [line.strip().split() for line in lines]
        yield lines


def tokens_from_file(tags_file):
    with codecs.open(tags_file, encoding='utf8') as inp:
        return [r.strip().split() for r in inp]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# WORKING: return tags directly
def extract_ter_alignment(hyps, refs, src_lang, trg_lang, tercom_path):

    tercom_jar = os.path.join(tercom_path, 'tercom.7.25.jar')

    output_prefix = os.path.join('{}-{}.tercom.out'.format(src_lang, trg_lang))

    # we need to put hyps and refs files in a special format
    # hyps_file_iter = codecs.open(hyps_file, encoding='utf8')
    # refs_file_iter = codecs.open(refs_file, encoding='utf8')
    # hyp_ref_iter = parallel_iterator(hyps_file_iter, refs_file_iter)

    hyp_ref_iter = parallel_iterator(hyps, refs)

    temp_hyps_file = tempfile.NamedTemporaryFile(delete=False)
    temp_refs_file = tempfile.NamedTemporaryFile(delete=False)
    for i, (hyp, ref) in enumerate(hyp_ref_iter):
        # Note the logic for escaping XML entities here
        temp_hyps_file.write(('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in hyp]), i)).encode('utf8'))
        temp_refs_file.write(('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in ref]), i)).encode('utf8'))
    temp_hyps_file.close()
    temp_refs_file.close()

    # with codecs.open(temp_hyps_file, 'w', encoding='utf8') as f_hyp:
    #     with codecs.open(temp_refs_file, 'w', encoding='utf8') as f_ref:
    #         for i, (hyp, ref) in enumerate(hyp_ref_iter):
    #             # Note the logic for escaping XML entities here
    #             f_hyp.write(('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in hyp]), i)).encode('utf8'))
    #             f_ref.write(('%s\t(%.12d)\n' % (u' '.join([cgi.escape(w) for w in ref]), i)).encode('utf8'))

    # Run TERCOM.
    cmd = 'java -jar {} -r {} -h {} -n {} -d 0'.format(tercom_jar,
                                                       temp_refs_file.name,
                                                       temp_hyps_file.name,
                                                       output_prefix)
    p = subprocess.Popen(cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()

    os.remove(temp_hyps_file.name)
    os.remove(temp_refs_file.name)

    # Parse TERCOM output xml
    mt_tokens, pe_tokens, edits, hters = \
        parse_pra_xml.parse_file('{}.xml'.format(output_prefix))

    tags_map = {'C': 'OK', 'S': 'BAD', 'I': 'BAD', 'D': 'BAD'}
    tags = [parse_pra_xml.get_tags(edit, tags_map, keep_inserts=False) for edit in edits]
    return tags

    # tags_output_file = os.path.join(output_path, output_prefix + '.tags')
    # with codecs.open(tags_output_file, 'w', encoding='utf8') as out:
    #     for row in tags:
    #         out.write(u' '.join(row) + u'\n')
    # logger.info('Wrote tags to: {}'.format(tags_output_file))
