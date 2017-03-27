import os
import errno
import itertools


def parallel_iterator(*iters):
    for lines in itertools.izip(*iters):
        lines = [line.strip().split() for line in lines]
        yield lines


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


