import codecs
import itertools


def parallel_iterator(src_iter, trg_iter):

    for src_l, trg_l in itertools.izip(src_iter, trg_iter):
        yield (src_l, trg_l)
