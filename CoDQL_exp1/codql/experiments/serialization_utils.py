import numpy as np
from copy import deepcopy

def index_serialization(serialization, idx):
    """indexing of the serialization"""
    
    frame = dict()
    def aux(f, s):
        for k in s:
            if type(s[k]) == dict:
                f[k] = dict()
                aux(f[k], s[k])
            else:
                f[k] = s[k][idx]

    aux(frame, serialization)
    return frame

def len_serialization(serialization):
    """len of serialization"""

    cur = serialization
    while True:
        if type(cur) == dict:
            cur = cur[list(cur.keys())[0]]
        else:
            return len(cur)
        # print(cur)
        # cur = cur[list(cur.keys())[0]]

def create_serialization(serialized_obj):
    """create a serialization that is formatted as serialized obj"""

    prototype = dict()

    def aux(p, d):
        for k in d:
            if type(d[k]) == dict:
                p[k] = dict()
                aux(p[k], d[k])
            else:
                p[k] = list()

    aux(prototype, serialized_obj)
    return prototype

def append_serialization(a, b):
    """append Serialization b to a"""
    def aux(x, y):
        for k in x:
            if type(x[k]) == dict:
                aux(x[k], y[k])
            else:
                x[k].append(deepcopy(y[k]))
    aux(a, b)

def flatten_nested_dict(old_dict, sep='-'):
    """flatten nested dict"""
    new_dict = dict()
    def aux(d, prefix=''):
        for k in d:
            if type(d[k]) == dict:
                aux(d[k], sep.join([prefix, k]))
            else:
                new_dict[sep.join([prefix, k])] = d[k]
    aux(old_dict)
    return new_dict

def unflatten_nested_dict(old_dict, sep='-'):
    """unflatten nested dict"""
    new_dict = dict()
    def add_components(paths, content):
        current = new_dict
        for path in paths:
            if path not in current:
                current[path] = dict()
            last = current
            current = current[path]
        last[path] = content

    for k in old_dict:
        paths = k.split(sep)
        add_components(paths, old_dict[k])

    return new_dict
    
def serialization_to_npz(a, filename):
    """savez serialization"""
    # print([a.keys()])
    # print([flatten_nested_dict(a).keys()])
    np.savez(filename, **flatten_nested_dict(a))

def npz_to_serialization(filename):
    """loadz serialization"""
    return unflatten_nested_dict(dict(np.load(filename)))['']
    

