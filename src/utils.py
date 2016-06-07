import os
import sys
import bz2
import pickle
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


def read_pkl_bz2(fname):
    with open(fname, 'rb') as f:
        data = f.read()
    compressed = bz2.decompress(data)
    return pickle.loads(compressed)


def get_line_num(path):
    num = sum(1 for _ in open(path))
    return num
