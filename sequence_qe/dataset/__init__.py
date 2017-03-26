import codecs
import itertools


def parallel_iterator(*iters):
    for lines in itertools.izip(*iters):
        lines = [line.strip().split() for line in lines]
        yield lines


