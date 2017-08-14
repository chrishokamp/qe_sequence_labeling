import logging
import os
import codecs
import re
from subprocess import Popen, PIPE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_pairs(word):
    """ (Subword Encoding) Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, cache=None):
    """
    (Subword Encoding) Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if cache is None:
        cache = {}

    if orig in cache:
        return cache[orig]

    word = tuple(orig) + ('</w>',)
    pairs = get_pairs(word)

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    cache[orig] = word
    return word


class BPE(object):

    def __init__(self, codes, separator='@@', ignore=None):
        self.bpe_codes = [tuple(item.split()) for item in codes]
        self.ignore = ignore

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])
        self.separator = separator

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = []
        for word in sentence.split():
            if self.ignore is not None and word in self.ignore:
                output.append(word)
            else:
                new_word = encode(word, self.bpe_codes)

                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])

        return u' '.join(output)


class DataProcessor(object):
    """
    This class encapusulates pre- and post-processing functionality

    """

    def __init__(self, lang, use_subword=False, subword_codes=None, escape_special_chars=False, truecase_model=None):
        self.use_subword = use_subword
        if self.use_subword:
            subword_codes_iter = codecs.open(subword_codes, encoding='utf-8')
            self.bpe = BPE(subword_codes_iter)

        self.lang = lang

        # Note hardcoding of script location within repo
        tokenize_script = os.path.join(os.path.dirname(__file__), 'resources/tokenizer/tokenizer.perl')
        self.tokenizer_cmd = [tokenize_script, '-l', self.lang, '-no-escape', '1', '-q', '-', '-b']
        self.tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE, bufsize=1)

        detokenize_script = os.path.join(os.path.dirname(__file__), 'resources/tokenizer/detokenizer.perl')
        self.detokenizer_cmd = [detokenize_script, '-l', self.lang, '-q', '-']

        self.escape_special_chars = escape_special_chars

        if self.escape_special_chars:
            escape_special_chars_script = os.path.join(os.path.dirname(__file__),
                                                       'resources/tokenizer/escape-special-chars.perl')
            self.escape_special_chars_cmd = [escape_special_chars_script]

            deescape_special_chars_script = os.path.join(os.path.dirname(__file__),
                                                         'resources/tokenizer/deescape-special-chars.perl')
            self.deescape_special_chars_cmd = [deescape_special_chars_script]

        self.truecase = False
        if truecase_model is not None:
            self.truecase = True

            truecase_script = os.path.join(os.path.dirname(__file__),
                                           'resources/recaser/truecase.perl')
            self.truecase_cmd = [truecase_script, '-m', truecase_model]

            detruecase_script = os.path.join(os.path.dirname(__file__),
                                             'resources/recaser/detruecase.perl')
            self.detruecase_cmd = [detruecase_script]

    def tokenize(self, text):
        if len(text.strip()) == 0:
            return []

        if type(text) is unicode:
            text = text.encode('utf8')
        self.tokenizer.stdin.write(text + '\n\n')
        self.tokenizer.stdin.flush()
        self.tokenizer.stdout.flush()

        # this logic is due to issues with calling out to the moses tokenizer
        segment = '\n'
        while segment == '\n':
            segment = self.tokenizer.stdout.readline()
        # read one more line
        _ = self.tokenizer.stdout.readline()
        segment = segment.rstrip()

        if self.escape_special_chars:
            char_escape = Popen(self.escape_special_chars_cmd, stdin=PIPE, stdout=PIPE)
            # this script cuts off a whitespace, so we add some extra
            segment, _ = char_escape.communicate(segment + '   ')
            segment = segment.rstrip()

        if self.truecase:
            # Chris: this takes forever, so commented until we find a realtime solution
            # truecaser = Popen(self.truecase_cmd, stdin=PIPE, stdout=PIPE)
            # this script cuts off a whitespace, so we add some extra
            # segment, _ = truecaser.communicate(segment + '   ')
            # segment = segment.rstrip()
            # Chris: hack which mocks truecasing
            segment = segment[0].lower() + segment[1:]

        utf_line = segment.decode('utf8')

        if self.use_subword:
            return self.bpe.segment(utf_line)
        else:
            return utf_line

    def detokenize(self, text):
        """
        Detokenize a string using the moses detokenizer

        Args:

        Returns:

        """
        if self.use_subword:
            text = re.sub("\@\@ ", "", text)
            text = re.sub("\@\@", "", text)

        if type(text) is unicode:
            text = text.encode('utf8')

        detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
        text, _ = detokenizer.communicate(text)

        utf_line = text.rstrip().decode('utf8')
        return utf_line

    def deescape_special_chars(self, text):
        if type(text) is unicode:
            text = text.encode('utf8')
        char_deescape = Popen(self.deescape_special_chars_cmd, stdin=PIPE, stdout=PIPE)
        # this script cuts off a whitespace, so we add some extra
        text, _ = char_deescape.communicate(text + '   ')
        text = text.rstrip()
        utf_line = text.decode('utf8')
        return utf_line

    def detruecase(self, text):
        if type(text) is unicode:
            text = text.encode('utf8')
        detruecaser = Popen(self.detruecase_cmd, stdin=PIPE, stdout=PIPE)
        # this script cuts off a whitespace, so we add some extra
        text, _ = detruecaser.communicate(text + '   ')
        text = text.rstrip()
        utf_line = text.decode('utf8')
        return utf_line
