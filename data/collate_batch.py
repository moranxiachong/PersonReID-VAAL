# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, new_labels, camids, viewids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    new_labels = torch.tensor(new_labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, viewids, new_labels


def val_collate_fn(batch):
    imgs, pids, new_labels, camids, viewids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

