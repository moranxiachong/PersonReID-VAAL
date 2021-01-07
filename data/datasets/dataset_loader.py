# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import pickle
import random


def restore_file(path):
    df = open(path, 'rb')
    file = pickle.load(df)
    df.close()
    return file


def read_image2(img_path, parts=False):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if parts == False:
                img = Image.open(img_path).convert('RGB')
            else:
                if random.random() <= 0.5:
                    img = Image.open(img_path).convert('RGB')
                else:
                    img = Image.open(img_path)
                    W,H = img.size
                    if random.random() <= 0.5:
                        if random.random() <= 0.0:
                            box = (0, 0, W, H//2)
                        else:
                            box = (0, 0, W, H//4*3)
                    else:
                        if random.random() <= 0.0:
                            box = (0, H//2, W, H)
                        else:
                            box = (0, H//4, W, H)
                    img = img.crop(box)
                    img = img.convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_image(img_path, parts=-1):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if parts == -1:
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path)
                W,H = img.size
                if parts == 0:
                    box = (0, 0, W, H//3)
                    #box = (0, 0, W, H//2)
                elif parts == 1:
                    box = (0, H//3, W, H//3*2)
                    #box = (0, H//2, W, H)
                elif parts == 2:
                    box = (0, H//3*2, W, H)
                else:
                    print("Error parts")
                    exit()
                img = img.crop(box)
                img = img.convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def image_map_new_label(dataset, image_map_direction):
    image_map_label2 = {}
    new_pid_set = set()
    for index in range(len(dataset)):
        img_path, pid, camid = dataset[index]
        viewid = image_map_direction[osp.basename(img_path)][0]
        new_pid_set.add((pid, viewid))
    print('** [number of new_pid]: ', len(new_pid_set))
    new_pid_list = list(new_pid_set)
    new_pid_list.sort()
    print(new_pid_list[:30])
    for i,j in enumerate(new_pid_list):
        image_map_label2[j] = i
    print('** [number of new_pid]: ', len(image_map_label2))
    return image_map_label2


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, path='', train=True, pcb_flag=False, parts=-1):
        self.dataset = dataset
        self.transform = transform
        self.train = train
        self.pcb = pcb_flag
        self.part = parts
        print('pcb_flag:', pcb_flag, ' parts:', self.part, ' train:', self.train)
        self.pcb_map = {0:0, 1:0, 2:1}
        #path = '/raid/home/zhihui/reid_strong_baseline/data/market1501/image_map_direction.pkl'
        self.image_map_direction = restore_file(path)
        if train:
            print('---------------------train-----------------------')
            self.image_map_label2 = image_map_new_label(dataset, self.image_map_direction)
            self.num_classes2 = len(self.image_map_label2)
        else:
            print('---------------------val+test-----------------------')

        print('**** ImageDataset init ok ...')
        print(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        if self.train:
            img = read_image(img_path, self.part)
        else:
            img = read_image(img_path, self.part)

        if self.train:
            if osp.basename(img_path) not in self.image_map_direction.keys():
                print('++++')
            viewid = self.image_map_direction[osp.basename(img_path)][0]
            new_label = self.image_map_label2[(pid, viewid)]
            if self.pcb:
                viewid = self.pcb_map[viewid]
        else:
            viewid = -1
            new_label = -1
        #print('+++', viewid)
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, new_label, camid, viewid, img_path
