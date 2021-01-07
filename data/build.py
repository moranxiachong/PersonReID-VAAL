# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms, build_transforms2, build_transforms3


def make_data_loader(cfg, c_fn=True):
    part = cfg.DATALOADER.PART
    if cfg.DATALOADER.TRANSFORM == 'init':
        train_transforms = build_transforms3(cfg, is_train=True)
        val_transforms = build_transforms3(cfg, is_train=False)
    elif cfg.DATALOADER.TRANSFORM == 'hard':
        train_transforms = build_transforms(cfg, is_train=True)
        val_transforms = build_transforms(cfg, is_train=False)
    elif cfg.DATALOADER.TRANSFORM == 'easy':
        train_transforms = build_transforms2(cfg, is_train=True)
        val_transforms = build_transforms2(cfg, is_train=False)
    else:
        print('Not found:', cfg.DATALOADER.TRANSFROM)
        exit(0)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        print(cfg.DATASETS.ROOT_DIR, 'fooo')
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        print(cfg.DATASETS.ROOT_DIR, 'fooo')
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms, cfg.DATASETS.PATH, train=True, parts=part)
    image_map_label2 = train_set.image_map_label2
    num_classes2 = train_set.num_classes2
    print('-- num_classes:',num_classes, '  num_classes2:', num_classes2)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        if cfg.DATASETS.ID_BALANCE == 'on':
            print('\n\n******* softmax with id balance \n\n')
            train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn)
        else:
            print('\n\n******* softmax without id balance \n\n')
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn)
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.DATASETS.PATH, train=False, parts=part)
    if c_fn:
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn)
    else:
        val_loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)
        train_loader = DataLoader(train_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(dataset.query), num_classes, num_classes2, image_map_label2
