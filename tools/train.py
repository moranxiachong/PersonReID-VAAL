# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn
import torch.optim as optim
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train#, do_train_with_center
from modeling import build_model
from layers import make_loss #, make_loss_with_center
from solver import make_optimizer, WarmupMultiStepLR #, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger
import functions
from torch.optim import lr_scheduler


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean)
    print(std)
    return mean, std


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, num_classes2, image_map_label2 = make_data_loader(cfg)

    #print('\n\n*** image_map_label2:')

    # prepare model
    model = build_model(cfg, num_classes, num_classes2)
    #print(list(model.children()))
    #print(model.state_dict().keys())
    #exit(0)

    #print('model.named_children(): \n\n', model.named_children())
    '''
    kk = 1
    for name, child in model.base.named_children():
        print(kk, name)
        kk += 1
    print(len(list(model.base.children())))
    exit(0)
    for i in range(len(list(model.base.children()))):
        print('  +++', i+1)
        print(list(model.base.children())[i])
    exit(0)
    '''

    if len(cfg.MODEL.PRETRAIN_PATH2)>5:
        print('--- resume from ', cfg.MODEL.PRETRAIN_PATH2)
        #model.load_param(cfg.MODEL.PRETRAIN_PATH)
        #model.loiad_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH2, map_location=lambda storage, loc: storage))
        if cfg.MODEL.ONCE_LOAD == 'yes':
            print('\n---ONCE_LOAD...\n')
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH2, map_location=lambda storage, loc: storage))
            #if cfg.MODEL.FREEZE_BASE == 'yes':
            #    functions.freeze_layer(model, 'base', False)
            #functions.freeze_global_model(model, False)
        else:
            functions.load_state_dict_distill(model, cfg.MODEL.PRETRAIN_PATH2, cfg.MODEL.ONLY_BASE, cfg.MODEL.WITHOUT_FC)
        print('**** Successfully load ', cfg.MODEL.PRETRAIN_PATH2)
        if cfg.MODEL.FREEZE_BASE:
            #functions.freeze_layer(model, 'base', False)
            functions.freeze_global_model(model, False)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        if cfg.SOLVER.MY_OPTIMIZER == "yes":
            print('---* my optimizer:', cfg.SOLVER.MY_OPTIMIZER_NAME)
            other_params = [p for n, p in model.named_parameters() if not n.startswith('base')]
            optimizer = optim.SGD([{'params': model.base.parameters(), 'lr': cfg.SOLVER.LR / 10},
                                   {'params': other_params, 'lr': cfg.SOLVER.LR}],
                                   momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            print('---* not my optimizer')
            optimizer = make_optimizer(cfg, model)

        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        #_C.SOLVER.MY_SCHEDULER = "no"
        #_C.SOLVER.MY_WARMUP = "no"
        loss_func = make_loss(cfg, num_classes)     # modified by gu

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            if cfg.SOLVER.MY_SCHEDULER == "yes":
                print('cfg.SOLVER.MY_SCHEDULER_STEP:', cfg.SOLVER.MY_SCHEDULER_STEP)
                print('---* my scheduler: ', cfg.SOLVER.MY_SCHEDULER_NAME)
                if cfg.SOLVER.MY_SCHEDULER_NAME == 'SL':
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.MY_SCHEDULER_STEP[0], gamma=0.1)
                elif cfg.SOLVER.MY_SCHEDULER_NAME == 'MSL':
                    scheduler = lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.MY_SCHEDULER_STEP, gamma=0.1)
                else:
                    print(cfg.SOLVER.MY_SCHEDULER_NAME, ' not found!')
                    eixt(0)
            else:
                print('---* not my scheduler')
                scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}

        print('************ do_train')
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,     # add for using self trained model
            image_map_label2,
            num_classes2
        )

#    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
#        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
#        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
#        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
#        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
#        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
#
#        arguments = {}
#
#        # Add for using self trained model
#        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
#            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
#            print('Start epoch:', start_epoch)
#            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
#            print('Path to the checkpoint of optimizer:', path_to_optimizer)
#            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
#            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
#            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
#            optimizer.load_state_dict(torch.load(path_to_optimizer))
#            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
#            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
#                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
#        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
#            start_epoch = 0
#            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
#                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
#        else:
#            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
#
#        do_train_with_center(
#            cfg,
#            model,
#            center_criterion,
#            train_loader,
#            val_loader,
#            optimizer,
#            optimizer_center,
#            scheduler,      # modify for using self trained model
#            loss_func,
#            num_query,
#            start_epoch     # add for using self trained model
#        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        device_id_str = ''
        for id in cfg.MODEL.DEVICE_ID:
            device_id_str += (str(id) + ',')
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id_str[:-1]    # new add by gu
    print('cfg.MODEL.DEVICE_ID: ', cfg.MODEL.DEVICE_ID)
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
