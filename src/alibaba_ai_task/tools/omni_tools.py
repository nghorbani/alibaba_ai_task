# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
import os
import os.path as osp
import random
import sys

import numpy as np
import torch


def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


def create_list_chunks(list_, group_size, overlap_size, cut_smaller_batches=True):
    if cut_smaller_batches:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size) if
                len(list_[i:i + group_size]) == group_size]
    else:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size)]


def trainable_params_count(params):
    return sum([p.numel() for p in params if p.requires_grad])


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_support_data_dir(current_fname=__file__):
    # print(current_fname)
    support_data_dir = osp.abspath(current_fname)
    support_data_dir_split = support_data_dir.split('/')
    # print(support_data_dir_split)
    try:
        support_data_dir = '/'.join(support_data_dir_split[:support_data_dir_split.index('src')])
    except:
        for i in range(len(support_data_dir_split)-1, 0, -1):
            support_data_dir = '/'.join(support_data_dir_split[:i])
            # print(i, support_data_dir)
            list_dir = os.listdir(support_data_dir)
            # print('-- ',list_dir)
            if 'support_data' in list_dir: break

    support_data_dir = osp.join(support_data_dir, 'support_data')
    assert osp.exists(support_data_dir)
    return support_data_dir

def id_generator(size=13):
    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def rm_spaces(in_text): return in_text.replace(' ', '_')