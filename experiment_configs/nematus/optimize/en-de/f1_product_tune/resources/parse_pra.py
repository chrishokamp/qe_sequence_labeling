import sys
import argparse

'''
Parse TERCOM .pra files
'''

def parse_sentence(line_array):
    hyp, ref = [], []
    align, sentence_id = "", ""
    for line in line_array:
        line_separator = line.find(':')
        line_id = line[:line_separator]
        if line_id == "Original Hyp":
            hyp = [w for w in line[line_separator+2:].split()]
        elif line_id == "Original Ref":
            ref = [w for w in line[line_separator+2:].split()]
        elif line_id == "Sentence ID":
            sentence_id = line[line_separator+2:]
        elif line_id == "Alignment":
            align = line[line_separator+3:-1]
            tags = []
            for ch in align:
                if ch == ' ':
                    tags.append('OK')
                else:
                    tags.append(ch)

    return (sentence_id, hyp, ref, tags)


def parse_file(file_name, fine_grained=False):
    instance = []
    mt_file = open(file_name+'.mt', 'w')
    pe_file = open(file_name+'.pe', 'w')
    tags_file = open(file_name+'.tags', 'w')
    if fine_grained:
        tags_map = {'OK': 'OK', 'S': 'BAD', 'I': 'BAD', 'D': 'BAD',
                    'SM': 'BAD_MV', 'IM': 'BAD_MV'}
    else:
        tags_map = {'OK': 'OK', 'S': 'BAD', 'I': 'BAD', 'D': 'BAD'}
    for line in open(file_name):
        if line == '\n':
            sent_id, hyp, ref, tags = parse_sentence(instance)
#            print(hyp, tags)
            assert(len(hyp) == len([t for t in tags if t != 'D'])), "Lengths mismatch: {} and {} in sentence {}".format(len(hyp), len([t for t in tags if t != 'D']), sent_id)
            assert(len(ref) == len([t for t in tags if t != 'I'])), "Lengths mismatch: {} and {} in sentence {}".format(len(ref), len([t for t in tags if t != 'I']), sent_id)

            if fine_grained:
                ref_tags = [t for t in tags if t != 'I']
                assert len(ref_tags) == len(ref)
                new_tags = list(tags)
                pos_hyp = 0
                pos_ref = 0
                for i, t in enumerate(tags):
                    if t in ['S', 'I']:
                        hyp_word = hyp[pos_hyp]
                        #if pos_ref >= len(ref):
                        #    import pdb
                        #    pdb.set_trace()
                        #ref_word = ref[pos_ref] # only use if t == 'S'
                        new_tags[i] = 'BAD'
                        if hyp_word in ref:
                            pos = ref.index(hyp_word)
                            if pos >= 0 and ref_tags[pos] != 'OK':
                                # Maybe this was caused by a reordering.
                                new_tags[i] += '_MV'
                        # Remove this.
                        if new_tags[i] == 'BAD':
                            if tags[i] == 'S':
                                new_tags[i] += '_SUB'
                            else:
                                new_tags[i] += '_DEL'
                        #if tags[i] == 'S' and new_tags[i] == 'BAD':
                        #    ref_word = ref[pos_ref]
                        #    new_tags[i] += '_%s_%s' % (hyp_word, ref_word)
                    elif t == 'D':
                        new_tags[i] = 'DEL'
                    if t == 'S' or t == 'OK':
                        pos_hyp += 1
                        pos_ref += 1
                    elif t == 'I':
                        pos_hyp += 1
                    else:
                        assert t == 'D'
                        pos_ref += 1
                tags = new_tags

            mt_file.write('%s\n' % (' '.join([w.encode('utf-8') for w in hyp])))
            pe_file.write('%s\n' % (' '.join([w.encode('utf-8') for w in ref])))
            if fine_grained:
                tags_file.write('%s\n' % (' '.join(\
                    [t.encode('utf-8') for t in tags if t != 'DEL'])))
            else:
                tags_file.write('%s\n' % (' '.join(\
                    [tags_map[t] for t in tags if t != 'D'])))
            instance = []
        instance.append(line[:-1].decode('utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tercom_file', type=str, required=True)
    parser.add_argument('-fine_grained', action='store_true')
    args = vars(parser.parse_args())
    filepath = args['tercom_file']
    fine_grained = args['fine_grained']
    print fine_grained
    parse_file(filepath, fine_grained)
