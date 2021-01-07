# coding=utf-8
import xml.etree.ElementTree as ET
import sys
import os
import glob
import shutil
import cv2
from multiprocessing import Pool
from multiprocessing import Manager
from multiprocessing import Process
import numpy as np
import pickle


def restore_file(path):
    df = open(path, 'rb')
    file = pickle.load(df)
    df.close()
    return file

def save_file(file, path, protocol=None):
    df = open(path, 'wb')
    if protocol is None:
        pickle.dump(file, df)
    else:
        pickle.dump(file, df, protocol=protocol)
    df.close()
    print('Successfully save ', path)


def get_direction(xml_path):
    tree = ET.parse(xml_path)
    rect={}
    line=""
    root = tree.getroot()
    #for name in root.iter('path'):
    #    rect['path'] = os.path.basename(name.text)
        
    def get_info(ob, name):
        for front in ob.iter(name):
            return int(front.text)

    for ob in root.iter('attributes'):
        rect['front'] = get_info(ob, 'front')
        rect['back'] = get_info(ob, 'back')
        rect['side'] = get_info(ob, 'side')
        rect['front_side'] = get_info(ob, 'front_side')
        rect['back_side'] = get_info(ob, 'back_side')
        rect['noise'] = get_info(ob, 'noise')
    try:
        sums = sum(rect.values())
    except:
        sums = 0
    return rect, sums


def mkdirs(root_dir):
    if os.path.exists(root_dir) is False:
        os.mkdir(root_dir)
    direction_list = ['front', 'back', 'side', 'front_side', 'back_side', 'noise', 'null', 'error']
    for i in direction_list:
        if os.path.exists(root_dir+i) is False:
            os.mkdir(root_dir+i)


def get_copy_list():
    save_dir = 'cuhk03_train_fixed2/'
    mkdirs(save_dir)
    xml_list = glob.glob('cuhk03_annotations/*.xml')
    copy_list = []
    print('len(xml_list):', len(xml_list))
    key_list = ['front', 'back', 'side', 'front_side', 'back_side', 'noise']
    num_dict = {}
    for i in key_list:
        num_dict[i] = 0


    for index, path in enumerate(xml_list):
        if index % 5000 == 0:
            print(index, len(xml_list))
        rect, sums = get_direction(path)
        if sums == 0:
            #shutil.copyfile(path, save_dir+'null/'+os.path.basename(path))
            copy_list.append([path, save_dir+'null/'+os.path.basename(path)])
            path1 = path.replace('.xml', '.jpg')
            #shutil.copyfile(path1, save_dir+'null/'+os.path.basename(path1))
            copy_list.append([path1, save_dir+'null/'+os.path.basename(path1)])
            continue
        if sums > 1:
            #shutil.copyfile(path, save_dir+'error/'+os.path.basename(path))
            copy_list.append([path, save_dir+'error/'+os.path.basename(path)])
            path1 = path.replace('.xml', '.jpg')
            #shutil.copyfile(path1, save_dir+'error/'+os.path.basename(path1))
            copy_list.append([path1, save_dir+'error/'+os.path.basename(path1)])
            continue
        for key in rect.keys():
            if rect[key] == 1:
                num_dict[key] += 1
                path1 = path.replace('.xml', '.jpg')
                #shutil.copyfile(path1, save_dir+key+'/'+os.path.basename(path1))
                copy_list.append([path1, save_dir+key+'/'+os.path.basename(path1)])
                break
    print('-------------')
    for i in key_list:
        print(i, num_dict[i], round(num_dict[i]/len(xml_list), 3))

    print('-------------')
    print(round((num_dict['front']+num_dict['front_side'])/len(xml_list), 3))
    print(round((num_dict['back']+num_dict['back_side'])/len(xml_list), 3))
    print(round((num_dict['side'])/len(xml_list), 3))

    return copy_list


def copy_img(path_list):
    for path in path_list:
        shutil.copyfile(path[0], path[1])


def split_direction():
    copy_list = get_copy_list()
    print('len(copy_list):', len(copy_list))

    #exit(0)
    num_jobs = 8
    index_list = len(copy_list)*np.arange(0,1,1/num_jobs)
    index_list = [int(i) for i in index_list]
    index_list.append(len(copy_list))
    print(index_list)

    processes = list()
    for i in range(num_jobs):
        p = Process(target=copy_img, args=(copy_list[index_list[i]:index_list[i+1]],))
        print('Process %d will start.' % i)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def get_bbox(xml_path):
    tree = ET.parse(xml_path)
    rect={}
    line=""
    root = tree.getroot()
    #for name in root.iter('path'):
    #    rect['path'] = os.path.basename(name.text)
        
    def get_info(ob, name):
        for front in ob.iter(name):
            return int(front.text)

    for ob in root.iter('bndbox'):
        #for obb in root.iter('bndbox'):
        xmin = get_info(ob, 'xmin')
        ymin = get_info(ob, 'ymin')
        xmax = get_info(ob, 'xmax')
        ymax = get_info(ob, 'ymax')
        break
    print(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


if __name__ == '__main__':
    '''
    name = 'wait_to_crop_train/0010_c6s4_002427_07.jpg'
    xmin, xmax, ymin, ymax = get_bbox('wait_to_crop_train/0010_c6s4_002427_07.xml')
    img = cv2.imread(name)
    #cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (255,0,0),1)
    img2 = img[ymin:ymax, xmin:xmax]
    cv2.imshow('image', img2)
    cv2.waitKey(0)
    exit(0)
    '''
    image_list = glob.glob('wait_to_crop_test/*.jpg')
    for name in image_list:
        basename = os.path.basename(name)
        img = cv2.imread(name)
        if os.path.exists('wait_to_crop_test/'+basename[:-4]+'.xml'):
            xmin, xmax, ymin, ymax = get_bbox('wait_to_crop_test/'+basename[:-4]+'.xml')
            img = cv2.imread(name)
            img2 = img[ymin:ymax, xmin:xmax]
            cv2.imwrite('crop_test/'+basename, img2)

    exit(0)
    #split_direction()

    image_map_direction = {}
    direction_map_image = {}
    img_list = []
    save_dir = 'cuhk03_train_fixed2/'
    direction_list = ['front', 'back', 'side', 'front_side', 'back_side', 'noise']

    map_int = {'front':0,
               'front_side': 0,
               'side':1,   
               'noise':1,
               'back': 2, 
               'back_side':2,}

    map_int2 = {'front':0,
               'front_side': 1,
               'side':2,   
               'noise':5,
               'back': 3, 
               'back_side':4,}


    direction_int_list = []
    direction_int_list2 = []
    for i in direction_list:
        image_list = os.listdir(save_dir+i)
        direction_map_image[i] = image_list
        for name in image_list:
            image_map_direction[name] = (map_int[i], i)
            direction_int_list.append(map_int[i])
            direction_int_list2.append(map_int2[i])
            if name[-8:] == '.jpg.jpg':
                image_map_direction[name[:-4]] = (map_int[i], i)
                print(name, name[:-4])

    print(len(direction_int_list), 
         round(direction_int_list.count(0)/len(direction_int_list), 2),
         round(direction_int_list.count(1)/len(direction_int_list), 2),
         round(direction_int_list.count(2)/len(direction_int_list), 2))
    print(set(direction_int_list))

    print(len(direction_int_list2), 
         round(direction_int_list2.count(0)/len(direction_int_list2), 2),
         round(direction_int_list2.count(1)/len(direction_int_list2), 2),
         round(direction_int_list2.count(2)/len(direction_int_list2), 2),
         round(direction_int_list2.count(3)/len(direction_int_list2), 2),
         round(direction_int_list2.count(4)/len(direction_int_list2), 2),
         round(direction_int_list2.count(5)/len(direction_int_list2), 2))
    print(set(direction_int_list2))
    

    save_file(image_map_direction, 'cuhk03_image_map_direction.pkl')
    save_file(direction_map_image, 'cuhk03_direction_map_image.pkl')

    save_file(image_map_direction, 'cuhk03_image_map_direction_py2.pkl', 2)
    save_file(direction_map_image, 'cuhk03_direction_map_image_py2.pkl', 2)

    print(len(image_map_direction))
    exit(0)
    print(image_map_direction)


    exit(0)
    image_map_direction = {}
    direction_map_image = {}
    save_dir = 'market1501_full_fixed2/'
    direction_list = ['front', 'back', 'side', 'front_side', 'back_side', 'noise', 'null', 'error']
    for i in direction_list:
        image_list = os.listdir(save_dir+i)


    exit(0)
    

    exit(0)
    #save_dir = 'DukeMTMC-reID_detail/'
    save_dir = 'DukeMTMC-reID_detail/'
    direction_list = ['front', 'back', 'side', 'front_side', 'back_side']
    for i in direction_list:
        listglob1 = glob.glob(save_dir+i+'/*.jpg')
        for path in listglob1:
            img = cv2.imread(path)
            img = cv2.resize(img, ((50,120)))
            cv2.imwrite(path, img)

    #line = rect['path'] + "\t"+ rect['xmin']+ "\t"+rect['ymin']+"\t"+rect['xmax']+"\t"+rect['ymax']
