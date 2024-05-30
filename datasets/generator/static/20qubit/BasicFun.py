import numpy as np
import torch as tc
import os
import pickle
import time
import copy


def choose_device(n=0):
    return tc.device("cuda:"+str(n) if tc.cuda.is_available() else "cpu")


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def mkdir(path):
    path = path.strip()
    path = path.rstrip('/')
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


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
        return False


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


def output_txt(x, filename='data'):
    np.savetxt(filename + '.txt', x)


def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', log=None):
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
    if log is None:
        print(express)
    else:
        fprint(express, log)
    return express


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


def output_data(x, y, fileName):
    for nn in range(len(x)):
        fprint('%g \t %.14g ' % (x[nn], y[nn]), fileName, append=(not nn == 0))


def combine_dicts(dic1, dic2):
    # dic1中的重复key值将被dic2覆盖
    return dict(dic1, **dic2)


def entanglement_entropy(lm, tol=1e-20):
    if type(lm) is tc.Tensor:
        lm = lm / lm.norm()
        lm = lm ** 2 + tol
        ent = -(lm * tc.log(lm)).sum().item()
    else:
        lm = lm / np.linalg.norm(lm)
        lm = lm ** 2 + tol
        ent = -np.sum(lm * np.log(lm))
    return ent
