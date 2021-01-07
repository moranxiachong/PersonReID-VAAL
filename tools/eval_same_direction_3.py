# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
from os import mkdir
import sys, time
import torch
from torch.backends import cudnn
import cv2

sys.path.append('.')
'''
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
#from utils.logger import setup_logger
import functions
'''
import pickle
import numpy as np
import random
import pandas as pd

def save_file(file, path):
    df = open(path, 'wb')
    pickle.dump(file, df)
    df.close()
    print('Successfully save ', path)


def restore_file(path):
    df = open(path, 'rb')
    file = pickle.load(df)
    df.close()
    print('Successfully restore from ', path)
    return file

class ShowProcess():
    i = 0
    max_steps = 0
    max_arrow = 100
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_names, g_names, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    #if name_list is not None:
    #    name_array = np.array(name_list)

    # compute cmc curve for each query
    all_cmc = []
    all_cmc2 = []
    all_AP = []
    all_gt_flag = []
    all_g_names = []
    g_names = np.array(g_names)
    num_valid_q = 0.  # number of valid query
    kkk = 0
    for q_idx in range(num_q):
        kkk += 1
        if kkk % 100 == 0:
            print(kkk, '/', num_q)
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_name_item = q_names[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        #print('keep:\n', keep[:100])
        #print('len(keep):', len(keep))

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        '''
        print('len(orig_cmc):', len(orig_cmc))
        print('len(matches[q_idx]):', len(matches[q_idx]))
        print('matches[q_idx]:\n', matches[q_idx][:30])
        print('orig_cmc:\n', orig_cmc[:30])
        '''
        g_names_order = g_names[order][keep]
        all_g_names.append(g_names_order[:50])
        '''
        print(g_names_order[:30])
        print(g_names_order[keep][:30])
        print(len(g_names_order), len(g_names_order[keep]))
        '''
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            print('No such query!!!')
            continue

        all_gt_flag.append(orig_cmc[:max_rank])
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        all_cmc2.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        #print(AP)
        '''
        print('query:', q_name_item)
        print('AP:', AP)
        print('CMC-1-10: ', tmp_cmc[:10])
        exit(0)
        '''
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    save_dir = 'view_result3/'
    print('+++ start view_result...')
    view_result(save_dir, q_names, g_names, all_g_names, all_AP, all_cmc2, all_gt_flag)
    print(sorted(all_AP))

    return all_cmc, mAP

def view_result(save_dir, q_names, g_names, all_g_names, all_AP, all_cmc, all_gt_flag):
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    '''
    df = pd.DataFrame({'q_names':list(q_names),
                       'all_g_names':all_g_names,
                       'all_AP':all_AP,
                       'all_cmc':all_cmc,
                       'all_gt_flag':all_gt_flag})
    df.to_csv(save_dir + 'result.csv', index=None)
    '''

    q_dir = '/raid/home/zhihui/reid_strong_baseline/data/market1501/query/'
    g_dir = '/raid/home/zhihui/reid_strong_baseline/data/market1501/bounding_box_test/'
    g_names_base = [os.path.basename(name) for name in g_names]  # 0006_c6s4_002202_00.jpg
    id_map_names = {}
    for name in g_names_base:
        if int(name.split('_')[0]) in id_map_names:
            id_map_names[int(name.split('_')[0])].append(name)
        else:
            id_map_names[int(name.split('_')[0])] = [name]

    for i in range(len(q_names)):
        AP, cmc = all_AP[i], all_cmc[i]
        gt_flag = all_gt_flag[i]
        q_name_item = os.path.basename(q_names[i])
        pid = int(q_name_item.split('_')[0])
        pid_img_num = len(id_map_names[pid])
        g_names_pred = [os.path.basename(name) for name in all_g_names[i]]
        g_names_gt = []
        for j in range(len(gt_flag)):
            if gt_flag[j]:
                g_names_gt.append(g_names_pred[j])
        g_names_gt = g_names_gt + list(set(id_map_names[pid])-set(g_names_pred))

        result = draw_result_case(q_name_item, g_names_pred, g_names_gt, q_dir, g_dir, gt_flag)
        save_path = save_dir + str(1000+int(100*AP))[1:]+'_'+q_name_item
        #cv2.imwrite(save_path, result)

def draw_result_case(q_name_item, g_names_pred, g_names_gt, q_dir, g_dir, gt_flag):
    cols = 23
    rows = len(g_names_gt)//20+3

    h = 256
    w = 128
    font = cv2.FONT_HERSHEY_SIMPLEX
    dst=np.zeros((rows*h,cols*w,3),np.uint8)

    # query
    img = cv2.imread(q_dir+q_name_item)
    img = cv2.resize(img, (w,h))
    cv2.putText(img, q_name_item.split('_')[0], (20, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(img, q_name_item.split('_')[1], (20, 60), font, 1, (0, 255, 0), 2)
    roi = img[0:h,0:w,:]
    dst[0:h, 0:w,:] = roi

    # g_names_pred
    for i in range(20):
        name = g_names_pred[i]
        img = cv2.imread(g_dir+name)
        img = cv2.resize(img, (w,h))
        cv2.putText(img, name.split('_')[0], (20, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(img, name.split('_')[1], (20, 60), font, 1, (0, 0, 255), 2)
        color = (0,255,0) if gt_flag[i] else (0,0,255)
        cv2.rectangle(img,(0,0),(w,h),color,5)
        roi = img[0:h, 0:w, :]
        row_index = 0
        col_index = i+2
        dst[row_index*h:(row_index+1)*h, col_index*w:(col_index+1)*w,:] = roi

    # g_names_gt
    for i in range(len(g_names_gt)):
        name = g_names_gt[i]
        img = cv2.imread(g_dir+name)
        img = cv2.resize(img, (w,h))
        cv2.putText(img, name.split('_')[0], (20, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(img, name.split('_')[1], (20, 60), font, 1, (0, 0, 255), 2)
        roi = img[0:h, 0:w, :]
        row_index = i//20+2
        col_index = i%20+2
        dst[row_index*h:(row_index+1)*h, col_index*w:(col_index+1)*w,:] = roi

    return dst

def save_single_track_image(root_dir, image_list, key_name):
    rows = len(image_list)//10+1
    cols = len(image_list)%10
    h = 256
    w = 128
    font = cv2.FONT_HERSHEY_SIMPLEX
    dst=np.zeros((rows*h,10*w,3),np.uint8)
    for i in range(len(image_list)):
        row_index = i//10
        col_index = i%10
        img = cv2.imread(root_dir+image_list[i])
        img = cv2.resize(img, (w,h))
        #cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
        roi=img[0:h,0:w,:]
        dst[row_index*h:(row_index+1)*h, col_index*w:(col_index+1)*w,:] = roi
    cv2.imwrite(key_name,dst)

def save_all_track_image(vid_dict, root_dir, save_dir):
    for i in vid_dict.keys():
        save_track_image(root_dir, vid_dict[i], save_dir+i+'.jpg')

def R1_mAP(num_query, feats, pids, camids, name_list, max_rank=50, feat_norm='yes'):
        #feats = torch.cat(feats, dim=0)
        if feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        q_names = np.asarray(name_list[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        g_names = np.asarray(name_list[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_names, g_names, max_rank=max_rank)

        return cmc, mAP

def extract_feature(model, val_loader, num_query, eval_flag=True):
    model.eval()
    feats_list, name_list, pid_list, camid_list = [], [], [], []

    process_bar = ShowProcess(len(val_loader), 'OK')
    for img, pid, new_label, camid, viewid, img_path in val_loader:
        with torch.no_grad():
            img = img.to(device) if torch.cuda.device_count() >= 1 else data
            feats = model(img)
            size = feats.shape
            feats = feats.view(size[0], -1)
            feats_list.append(feats.cpu().data)
            name_list += list(img_path)
            pid_list += list(pid)
            camid_list += list(camid)
            process_bar.show_process()

    test_features = torch.cat(feats_list, dim=0)
    pid_list = np.array(pid_list)
    camid_list = np.array(camid_list)
    name_list = np.array(name_list)
    if eval_flag:
        cmc, mAP = R1_mAP(num_query, test_features, pid_list, camid_list, max_rank=50, feat_norm='yes')

        print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))

        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])

    return test_features, pid_list, camid_list, name_list, ss

def split_query_gallery(front_list):
    id_map_images = {}
    select_id_map_images = {}
    query_list, gallery_list = [], []
    for name in front_list:
        idx = int(name.split('_')[0])
        if idx in id_map_images:
            id_map_images[idx].append(name)
        else:
            id_map_images[idx] = [name]

    nums = 0
    for key in id_map_images.keys():
        if len(id_map_images[key])>1:
            nums += 1
            select_id_map_images[key] = id_map_images[key]
            a = id_map_images[key]
            random.shuffle(a)
            query_list.append(a[0])
            gallery_list += a[1:]

    print(' -- select id numbers:', nums)
    return query_list, gallery_list

def select_features(query_list, name_list, feats, pid_list, camid_list):
    name_list2 = [name.split('/')[-1] for name in name_list]
    query_list2 = []
    len1 = len(set(name_list2))
    for i in range(len(name_list2)):
        if name_list2[i][-8:]=='.jpg.jpg':
            #print(name_list2[i])
            name_list2[i] = name_list2[i][:-4]
            #print(name_list2[i])
    len2 = len(set(name_list2))
    if len1 != len2:
        print('Error ... name_list2 length')

    len1 = len(set(query_list))
    for i in range(len(query_list)):
        if query_list[i][-8:]=='.jpg.jpg':
            #print(query_list[i])
            query_list2.append(str(query_list[i][:-4]))
        else:
            query_list2.append(str(query_list[i]))
    len2 = len(set(query_list2))
    if len1 != len2:
        print('Error ... query_list2 length')

    name_map_feat = {}
    name_map_pid = {}
    name_map_camid = {}
    #print(query_list[:3])
    #print(name_list2[:3])
    for i in range(len(name_list2)):
        name_map_feat[name_list2[i]] = feats[i].view(1,-1)
        name_map_pid[name_list2[i]] = pid_list[i]
        name_map_camid[name_list2[i]] = camid_list[i]
    query_feat_list = [name_map_feat[str(name)] for name in query_list2]
    query_pid_list = [name_map_pid[name] for name in query_list2]
    query_camid_list = [name_map_camid[name] for name in query_list2]

    #print(len(query_feat_list))
    #print(query_feat_list[0].shape)
    query_feat = torch.cat(query_feat_list, dim=0)
    #print('query_feat.shape:', query_feat.shape)
    return query_feat, query_pid_list, query_camid_list

def avg_list(cmc_list):
    result_list = cmc_list[0]
    for i in range(1, len(cmc_list)):
        result_list += cmc_list[i]
    result_list = result_list/len(cmc_list)
    return result_list

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="configs/se_resnext101_softmax_triplet.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    '''

    #cudnn.benchmark = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    '''
    h = 256
    w = 128
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    name = '0002_c1s1_000551_01.jpg'
    img = cv2.imread('Market-1501-fixed/bounding_box_train/0002_c1s1_000551_01.jpg')
    img = cv2.resize(img, (w,h))
    cv2.putText(img, name.split('_')[0], (20, 40), font, 1, (0, 0, 255), 1)
    cv2.putText(img, name.split('_')[1], (20, 70), font, 1, (0, 0, 255), 1)
    cv2.rectangle(img,(0,0),(128,256),(0,255,0),5)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    df = pd.DataFrame({'a':[1,2,3],
          'b':['a','c','f']})
    df.to_csv('a.csv', index=None)
    exit(0)
    #roi=img[0:h,0:w,:]
    '''


    model_info_128 = [('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_0_2048_sample_softmax/net_170.pth', 0),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_1_2048_sample_softmax/net_140.pth', 1),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_2_2048_sample_softmax/net_175.pth', 2),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_no_2048_sample_softmax_H128_W128/net_190.pth', -1)]

    model_info_224 = [('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_0_2048_sample_softmax_H224_W224/net_150.pth', 0),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_1_2048_sample_softmax_H224_W224/net_70.pth', 1),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_2_2048_sample_softmax_H224_W224/net_190.pth', 2),
                      ('/data2/zhihui/my/reid_strong_baseline/logs/se_resnext101_Adam_part_no_2048_sample_softmax_H224_W224/net_185.pth', -1)]

    model_norm = [('/raid/home/zhihui/reid_strong_baseline/logs/two_branch_se_resnext101_Adma_ep200_LR3_finetune_best_once_load/net_135.pth', -1)]
    model_base = [('/raid/home/zhihui/reid_strong_baseline/logs/se_resnext101_lsr_Adam_H386_W128/net_100.pth', -1)]
    model_base2 = [('/raid/home/zhihui/reid_strong_baseline/logs/softmax_se_resnext101/se_resnext101_model_160.pth', -1)]

    result_best = restore_file('features/market_two_branch_best_result.pkl')
    result_baseline = restore_file('features/market_two_branch_lsr_baseline_result.pkl')

    feats_best, pid_list_best, camid_list_best, name_list_best, num_query_best = result_best['features'], result_best['pids'], result_best['camids'], result_best['names'], result_best['num_query']
    feats_baseline, pid_list_baseline, camid_list_baseline, name_list_baseline, num_query_baseline = result_baseline['features'], result_baseline['pids'], result_baseline['camids'], result_baseline['names'], result_baseline['num_query']
    front_list = restore_file('pkl/market_mixQueryGallery_same_direction_front_list_py2.pkl')
    back_list = restore_file('pkl/market_mixQueryGallery_same_direction_back_list_py2.pkl')
    side_list = restore_file('pkl/market_mixQueryGallery_same_direction_side_list_py2.pkl')

    market_name = restore_file('pkl/market_name_py2.pkl')
    query_gallery_list, image_map_direction = market_name['query_gallery_list'], market_name['image_map_direction']

    #cmc, mAP = R1_mAP(len(query_list), mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline, max_rank=25, feat_norm='yes')
    cmc, mAP = R1_mAP(num_query_best, feats_best, pid_list_best, camid_list_best, name_list_best, max_rank=50, feat_norm='yes')
    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))

    exit(0)

    ff = open('tools/eval_same_direction_num5.txt', 'w')
    ff.write('\n\n\n ***** front_list:\n')
    print('\n\n\n ***** front_list:')
    test_num = 5
    cmc_list_best, mAP_list_best, cmc_list_baseline, mAP_list_baseline = [], [], [], []
    for i in range(test_num):
        print('   No.',i+1, ' /', test_num)
        query_list, gallery_list = split_query_gallery(front_list)
        print('       --> 1. best model')
        ff.write('       --> 1. best model\n')
        mix_feat_best, mix_pid_list_best, mix_camid_list_best = select_features(query_list+gallery_list, name_list_best,
                                                                                feats_best, pid_list_best, camid_list_best)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_best, mix_pid_list_best, mix_camid_list_best, max_rank=25, feat_norm='yes')
        cmc_list_best.append(cmc)
        mAP_list_best.append(mAP)
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        print(ss)
        ff.write(ss+"\n")

        print('       --> 2. baseline model')
        ff.write('       --> 2. baseline model\n')
        mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline = select_features(query_list+gallery_list, name_list_baseline,
                                                                        feats_baseline, pid_list_baseline, camid_list_baseline)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline, max_rank=25, feat_norm='yes')
        cmc_list_baseline.append(cmc)
        mAP_list_baseline.append(mAP)
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        print(ss)
        ff.write(ss+"\n")
    avg_cmc_best = avg_list(cmc_list_best)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_best)/len(mAP_list_best),
                                                   avg_cmc_best[0], avg_cmc_best[4], avg_cmc_best[9], avg_cmc_best[19])
    ff.write(ss+"\n")
    print(ss)
    avg_cmc_baseline = avg_list(cmc_list_baseline)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_baseline)/len(mAP_list_baseline),
                                                   avg_cmc_baseline[0], avg_cmc_baseline[4], avg_cmc_baseline[9], avg_cmc_baseline[19])
    ff.write(ss+"\n")
    print(ss)


    ff.write('\n\n\n ***** back_list:\n')
    print('\n\n\n ***** back_list:')
    #test_num = 10
    cmc_list_best, mAP_list_best, cmc_list_baseline, mAP_list_baseline = [], [], [], []
    for i in range(test_num):
        print('   No.',i+1, ' /', test_num)
        query_list, gallery_list = split_query_gallery(back_list)
        print('       --> 1. best model')
        ff.write('       --> 1. best model\n')
        mix_feat_best, mix_pid_list_best, mix_camid_list_best = select_features(query_list+gallery_list, name_list_best,
                                                                                feats_best, pid_list_best, camid_list_best)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_best, mix_pid_list_best, mix_camid_list_best, max_rank=25, feat_norm='yes')
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        cmc_list_best.append(cmc)
        mAP_list_best.append(mAP)
        print(ss)
        ff.write(ss+"\n")

        print('       --> 2. baseline model')
        ff.write('       --> 2. baseline model\n')
        mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline = select_features(query_list+gallery_list, name_list_baseline,
                                                                        feats_baseline, pid_list_baseline, camid_list_baseline)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline, max_rank=25, feat_norm='yes')
        cmc_list_baseline.append(cmc)
        mAP_list_baseline.append(mAP)
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        print(ss)
        ff.write(ss+"\n")
    avg_cmc_best = avg_list(cmc_list_best)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_best)/len(mAP_list_best),
                                                   avg_cmc_best[0], avg_cmc_best[4], avg_cmc_best[9], avg_cmc_best[19])
    ff.write(ss+"\n")
    print(ss)
    avg_cmc_baseline = avg_list(cmc_list_baseline)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_baseline)/len(mAP_list_baseline),
                                                   avg_cmc_baseline[0], avg_cmc_baseline[4], avg_cmc_baseline[9], avg_cmc_baseline[19])
    ff.write(ss+"\n")
    print(ss)


    ff.write('\n\n\n ***** side_list:\n')
    print('\n\n\n ***** side_list:')
    #test_num = 10
    cmc_list_best, mAP_list_best, cmc_list_baseline, mAP_list_baseline = [], [], [], []
    for i in range(test_num):
        print('   No.',i+1, ' /', test_num)
        query_list, gallery_list = split_query_gallery(side_list)
        print('       --> 1. best model')
        ff.write('       --> 1. best model\n')
        mix_feat_best, mix_pid_list_best, mix_camid_list_best = select_features(query_list+gallery_list, name_list_best,
                                                                                feats_best, pid_list_best, camid_list_best)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_best, mix_pid_list_best, mix_camid_list_best, max_rank=25, feat_norm='yes')
        cmc_list_best.append(cmc)
        mAP_list_best.append(mAP)
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        print(ss)
        ff.write(ss+"\n")

        print('       --> 2. baseline model')
        ff.write('       --> 2. baseline model\n')
        mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline = select_features(query_list+gallery_list, name_list_baseline,
                                                                        feats_baseline, pid_list_baseline, camid_list_baseline)
        cmc, mAP = R1_mAP(len(query_list), mix_feat_baseline, mix_pid_list_baseline, mix_camid_list_baseline, max_rank=25, feat_norm='yes')
        cmc_list_baseline.append(cmc)
        mAP_list_baseline.append(mAP)
        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])
        print(ss)
        ff.write(ss+"\n")
    avg_cmc_best = avg_list(cmc_list_best)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_best)/len(mAP_list_best),
                                                   avg_cmc_best[0], avg_cmc_best[4], avg_cmc_best[9], avg_cmc_best[19])
    ff.write(ss+"\n")
    print(ss)
    avg_cmc_baseline = avg_list(cmc_list_baseline)
    ss = "average best: mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(sum(mAP_list_baseline)/len(mAP_list_baseline),
                                                   avg_cmc_baseline[0], avg_cmc_baseline[4], avg_cmc_baseline[9], avg_cmc_baseline[19])
    ff.write(ss+"\n")
    print(ss)


    ff.close()

    # market_name = {'query_gallery_list':query_gallery_list,
    #               'query_list':query_list,
    #               'gallery_list':gallery_list,
    #               'train_list':train_list,
    #               'image_map_direction':image_map_direction}


    '''
    for item in model_base2:
        PRETRAIN_PATH, part = item

        train_loader, val_loader, num_query, num_classes, num_classes2, image_map_label2 = make_data_loader(cfg, c_fn=False)#, part_flag=True, part_value=part)

        model = build_model(cfg, num_classes, num_classes2)
        print('--- resume from ', PRETRAIN_PATH)

        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=lambda storage, loc: storage))

        #if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)
        model = model.to(device)

        print(item)
        test_features_norm, pid_list, camid_list, name_list, ss = extract_feature(model, val_loader, num_query)
        if part == -1:
            part = 'no'
        result = {'features':test_features_norm,
                  'pids':pid_list,
                  'camids':camid_list,
                  'names':name_list,
                  'num_query':num_query}
        save_file(result, 'features/market_two_branch_softmax_baseline_result.pkl')
        txt_name = 'features/market_two_branch_softmax_baseline_result.txt'
        #txt_name = 'features/result_three_part_%s_softmax_H224_W224.txt' % str(part)
        with open(txt_name, 'w') as f:
            f.write(ss)

        torch.cuda.empty_cache()
    '''

