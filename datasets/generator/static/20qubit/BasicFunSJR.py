import logging
import math
import os
import time
import pickle
import copy
from inspect import stack
from math import factorial

import numpy as np
import torch as tc
from termcolor import cprint

"""
List of functions:
arrangement
choose_device
combination
combine_dicts
fprint
kron
load
load_pr
now
output_txt
print_dict
project_path
save
save_pr
trace_stack
warning
"""


def choose_device(n=0):
    if n == 'cpu':
        return 'cpu'
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:"+str(n))
            else:
                return tc.device("cuda:"+str(n)[5:])
        else:
            return tc.device("cpu")


def project_path(project='T-Nalg/'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(project) + len(project)]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def save_pr(path, file, data, names):
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
            else:
                return None
    else:
        return None


def load_pr(path_file, names=None):
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
                s.close()
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def combine_dicts(dic1, dic2, deep_copy=False):
    # dic1中的重复key值将被dic2覆盖
    if deep_copy:
        return dict(dic1, **copy.deepcopy(dic2))
    else:
        return dict(dic1, **dic2)


def output_txt(x, filename='data'):
    if type(x) is not np.ndarray:
        x = np.array(x)
    np.savetxt(filename + '.txt', x)


def print_dict(a, keys=None, welcome='', style_sep=': ', color='white', end='\n'):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    cprint(express, color)
    return express


def print_progress_bar(n, nt, message=''):
    x1 = math.floor(n / nt * 10)
    x2 = math.floor(n / nt * 100) % 10
    if x1 == 10:
        message += chr(9646) * x1
    else:
        message += chr(9646) * x1 + str(x2) + chr(9647) * (9 - x1)
    print('\r'+message, end='')
    time.sleep(0.01)


def now(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    return time.strftime(fmt, time.localtime(time.time()))


def kron(mat1, mat2):
    return tc.einsum('ab,cd->acbd', mat1, mat2).reshape(
        mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])


def expm(mat, order=10):
    mat_f = tc.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    _mat = mat * 1.0
    for n in range(1, order+1):
        mat_f = mat_f + _mat
        _mat = _mat.mm(mat) / (n+1)
    return mat_f


def arrangement(n, m):
    return factorial(n) / factorial(n-m)


def combination(n, m):
    return int(arrangement(n, m) / factorial(m))


def random_samples_from_p(p, num):
    """
    :param p: 概率分布[p1, p2, ...]，满足p1+p2+...=1
    """
    y = [sum(p[:n]) for n in range(1, len(p)+1)]
    y = np.array(y)
    r = np.random.rand(num)
    ind = [np.nonzero(y > r[n])[0][0] for n in range(num)]
    return ind


def trace_stack(level0=2):
    # print the line and file name where this function is used
    info = stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def warning(string, if_trace_stack=False):
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)



