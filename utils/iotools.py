import os
import os.path as osp
import errno
import json
import shutil

import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best=False, f_path='checkpoint.ckp'):
    f_dir = osp.dirname(f_path)
    if len(f_dir) != 0:
        mkdir_if_missing(f_dir)
    if is_best:
        # shutil.copy(f_path, osp.join(f_dir, 'best_model.ckp'))
        f_path = osp.join(f_dir, 'best_model.ckp')
    torch.save(state, f_path)
