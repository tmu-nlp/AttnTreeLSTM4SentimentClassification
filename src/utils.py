import os
import sys
from progressbar import ProgressBar, Counter, Bar, Percentage
import chainer
import numpy as np


def num2onehot(num):
    return chainer.Variable(np.array([num], np.int32))


def zeros(size):
    return chainer.Variable(np.zeros((1, size), dtype=np.float32))


def get_src_path():
    path = __file__
    path = os.path.abspath(path)
    path = os.path.dirname(path)
    return path


def get_data_path():
    src = get_src_path()
    relative = '../data'
    path = os.path.join(src, relative)
    path = os.path.normpath(path)
    return path


def progressbar(iterable, maxval, message=' ', nested=False, nest_num=1):
    if message != ' ':
        message = ' "{}"'.format(message)
    wid = [Percentage(), ' ', Counter(), '/{}'.format(maxval), message, Bar()]
    pbar = ProgressBar(maxval=maxval, widgets=wid)
    if nested:
        sys.stdout.write('\n')
    for i, item in enumerate(iterable):
        pbar.update(pbar.value + 1)
        sys.stdout.flush()
        yield item
    pbar.finish()
    sys.stdout.flush()
    if nested:
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\r')
        sys.stdout.flush()
    for _ in range(nest_num - 1):
        sys.stdout.write('\n')


def get_line_num(path):
    num = sum(1 for _ in open(path))
    return num
