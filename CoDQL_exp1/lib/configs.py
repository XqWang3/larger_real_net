from __future__ import (absolute_import, division, print_function, unicode_literals)

from lib.collections import AttrDict
import six
import yaml
import copy
import logging
import numpy as np
from ast import literal_eval
import os.path as osp

cfg = None

def yaml_load(cfg_filename, *args, **kwargs):
    with open(cfg_filename, 'r') as f:
        d = yaml.load(f)
        d.update(*args, **kwargs)
        return AttrDict(d)

def dir_base(filepath):
    return osp.dirname(filepath), osp.basename(filepath)

def add_suffix(filename, suf='.yaml'):
    return filename + ('' if filename[-len(suf):]==suf else suf)

def load_cfg(cfg_filename):
    global cfg
    cfg = _load_cfg(cfg_filename)
    return cfg

def _load_cfg(cfg_filename):
    
    d, b = dir_base(cfg_filename)
    fp = osp.join(d, add_suffix(b))
    cfg = yaml_load(fp)
    dup = copy.deepcopy(cfg)

    n_par = cfg.PARENT if hasattr(cfg, "PARENT") else None
    l_modules = cfg.MODULES if hasattr(cfg, "MODULES") else []

    if n_par is not None:
        par = _load_cfg(osp.join(d, n_par))
        _merge_a_into_b(cfg, par)
        cfg = par

    for m in l_modules:
        mod = _load_cfg(osp.join(d, m))
        _merge_a_into_b(mod, cfg)
    
    return cfg

def merge(a, b):
    """Merge config dictionary a into config dictionary b
    """

    _merge_a_into_b(a, b)

def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():

        # preserved keywords
        if k == "PARENT" or k == "MODULES":
            continue

        full_key = '.'.join(stack) + '.' + k if stack is not None else k

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            b[k] = v
            continue
            raise KeyError('Non-existent config key: {}'.format(full_key))
        
        # AttrDict-ize the dict
        if type(b[k]) is dict:
            b[k] = AttrDict(b[k])

        # v = copy.deepcopy(v_)
        # v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict) and b[k] is not None:
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list, None
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif value_b is None:
        pass
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a