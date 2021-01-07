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

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
#from utils.logger import setup_logger
import functions
import pickle
import numpy as np


def save_file(file, path):
    df = open(path, 'wb')
    pickle.dump(file, df)
    df.close()
    print('Successfully save ', path)


def restore_file(path):
    df = open(path, 'rb') 
    file = pickle.load(df)
    df.close()
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

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def R1_mAP(num_query, feats, pids, camids, max_rank=50, feat_norm='yes'):
        #feats = torch.cat(feats, dim=0)
        if feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


def extract_feature(model, val_loader, num_query):
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
    cmc, mAP = R1_mAP(num_query, test_features, pid_list, camid_list, max_rank=50, feat_norm='yes')

    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))

    return test_features, pid_list, camid_list, name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # norm
    train_loader, val_loader, num_query, num_classes, num_classes2, image_map_label2 = make_data_loader(cfg, -1, False)


    model = build_model(cfg, num_classes, num_classes2)
    print('--- resume from ', cfg.MODEL.PRETRAIN_PATH2)

    model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH2, map_location=lambda storage, loc: storage))

    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model = model.to(device)
    
    # up
    train_loader, val_loader_up, num_query, num_classes, num_classes2, image_map_label2 = make_data_loader(cfg, 0, False)
    # down
    train_loader, val_loader_down, num_query, num_classes, num_classes2, image_map_label2 = make_data_loader(cfg, 1, False)

    print('\n ++ norm')
    test_features_norm, pid_list, camid_list, name_list = extract_feature(model, val_loader, num_query)
    result = {'features':test_features_norm, 
              'pids':pid_list, 
              'camids':camid_list, 
              'names':name_list,
              'num_query':num_query}
    #save_file(result, 'features/result_norm.pkl')

    print('\n ++ up')
    test_features_up, pid_list, camid_list, name_list = extract_feature(model, val_loader_up, num_query)
    result = {'features':test_features_up, 
              'pids':pid_list, 
              'camids':camid_list, 
              'names':name_list,
              'num_query':num_query}
    #save_file(result, 'features/result_up.pkl')

    print('\n ++ down')
    test_features_down, pid_list, camid_list, name_list = extract_feature(model, val_loader_down, num_query)
    result = {'features':test_features_down, 
              'pids':pid_list, 
              'camids':camid_list, 
              'names':name_list,
              'num_query':num_query}
    #save_file(result, 'features/result_down.pkl')


    cmc, mAP = R1_mAP(num_query, torch.cat((test_features_up, test_features_down), dim=1), pid_list, camid_list, max_rank=50, feat_norm='yes')
    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))


    cmc, mAP = R1_mAP(num_query, torch.cat((test_features_norm, test_features_up, test_features_down), dim=1), pid_list, camid_list, max_rank=50, feat_norm='yes')
    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))


    cmc, mAP = R1_mAP(num_query, torch.cat((test_features_norm, test_features_up), dim=1), pid_list, camid_list, max_rank=50, feat_norm='yes')
    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))


    cmc, mAP = R1_mAP(num_query, (test_features_norm+test_features_up+test_features_down)/3, pid_list, camid_list, max_rank=50, feat_norm='yes')
    print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))


    torch.cuda.empty_cache()






    

    
