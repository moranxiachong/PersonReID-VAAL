import os
import torch
import numpy as np
from torch.nn import Parameter


def load_state_dict(model, ckpt_file, only_base='no', without_fc='no'):
    print('\n\n\n')
    src_state_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    #target_dict_name_list = list(model.state_dict().keys())
    #base_dict_name_list = [p for n in target_dict_name_list if n.startswith('base')]
    #other_params = [p for n in target_dict_name_list if not n.startswith('base')]

    if only_base == 'yes':
        print('-- only backbone')
    if without_fc == 'yes':
        print('-- without FC')
    dest_state_dict = model.state_dict()
    print('\n---- dest_state_dict.keys():')
    list1 = dest_state_dict.keys()
    list2 = []
    set_des = set()
    for name in list1:
        if name.startswith('base'):
            set_des.add(name.split('.')[0])
            continue
        else:
          list2.append(name)
    print(list2)


    print('\n---- src_state_dict.keys():')
    list1 = src_state_dict.keys()
    list2 = []
    set_src = set()
    for name in list1:
        if name.startswith('base'):
            set_src.add(name.split('.')[0])
            continue
        else:
          list2.append(name)
    print(list2)

    print('set_des:', set_des)
    print('set_src:', set_src)

    if set_des == set_src:
        same_base = True
    else:
        same_base = False
    print('same_base:', same_base)

    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            print('!!! not find: ', name)
            continue
        if only_base == 'yes':
            if not name.startswith('base'):
                continue
        if without_fc == 'yes':
            if name.startswith('fc') or name.startswith('classifier') :
                continue

        if isinstance(param, Parameter):
            param = param.data
        try:
            if same_base:
                if name.startswith('classifier'):
                    dest_state_dict[name].copy_(param)
                    name_copy = name.replace('classifier', 'classifier_max')
                    dest_state_dict[name_copy].copy_(param)
                elif name.startswith('bottleneck'):
                    dest_state_dict[name].copy_(param)
                    name_copy = name.replace('bottleneck', 'bottleneck_max')
                    dest_state_dict[name_copy].copy_(param)
                else:
                    dest_state_dict[name].copy_(param)
            else:
                if name.startswith('base.'):
                    for name_item in set_des:
                        name_copy = name.replace('base', name_item)
                        dest_state_dict[name_copy].copy_(param)
                else:
                    dest_state_dict[name].copy_(param)
                    #if name.startswith('classifier.'):
                    #    dest_state_dict[name.replace('classifier', 'classifier_2')].copy_(param)
                    if name.startswith('bottleneck'):
                        dest_state_dict[name.replace('bottleneck', 'bottleneck_2')].copy_(param)
        except:
            msg = 'not found'
            print("Warning: Error occurs when copying '{}': {}".format(name, msg))

    return 0
    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("*****  Keys not found in source state_dict: ")
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        for n in dest_missing:
            print('\t', n)

    #exit(0)


def freeze_layer(model, layers=None, value=True):
    print('\n\nfreeze_layer: ', layers, value)
    if layers == 'base' or layers == 'classifier' or layers == 'classifier2':
        pass
    else:
        print('Not found! ', layers)
        exit(0)

    if value == True or value == False:
        pass
    else:
        print('Not found! ', value)
        exit(0)

    for name, child in model.named_children():
        if name == layers:
            print('+++ start handle ...')
            for param in child.parameters():
                param.requires_grad = value
    print('+++ end handle !!!')


def freeze_global_model(model, value=True):
    print('\n\nfreeze_global_model: ', value)

    if value == True or value == False:
        pass
    else:
        print('Not found! ', value)
        exit(0)

    name_set = {'base', 'gap', 'bottleneck', 'classifier', 'classifier2'}

    print('+++ start handle ...')
    for name, child in model.named_children():
        if name in name_set:
            print(name)
            for param in child.parameters():
                param.requires_grad = value
    if value:
        print('model.bottleneck.bias')
        model.bottleneck.bias.requires_grad_(False)

    print('+++ end handle !!!')
