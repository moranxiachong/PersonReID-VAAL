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


def extract_feature(model, val_loader, num_query, eval_flag=True):
    model.eval()
    feats_list, name_list, pid_list, camid_list = [], [], [], []
    #for mm in val_loader:
    #    print(len(mm))
    #    print(mm)

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

    np.save('market_resnet101_test_names.npy', name_list)
    np.save('market_resnet101_test_feats.npy', np.array(test_features))

    if eval_flag:
        cmc, mAP = R1_mAP(num_query, test_features, pid_list, camid_list, max_rank=50, feat_norm='yes')

        print("mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19]))

        ss = "mAP: {:.2%}    Rank-1:{:.2%}  Rank-5:{:.2%}  Rank-10:{:.2%}  Rank-20:{:.2%}".format(mAP,cmc[0],cmc[4],cmc[9],cmc[19])

    return test_features, pid_list, camid_list, name_list, ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="configs/eval_part_se_resnext101.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    files = restore_file('/raid/home/zhihui/reid_strong_baseline/features/market_two_branch_lsr_baseline_result.pkl')
    feats = np.array(files['features'])
    names = np.array(files['names'])
    np.save('market_baseline_lsr_test_feats.npy', feats)
    np.save('market_baseline_lsr_test_names.npy', names)
    print(feats.shape)
    exit(0)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

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
    model_base3 = [('/raid/home/zhihui/reid_strong_baseline/logs/fixed_market_se_resnext101_Adam_init_distill_no_finetune_layer4_bz96_true_fixed/net_225.pth', -1)]
    model_base4 = [('/raid/home/zhihui/reid_strong_baseline/logs/Duke_se_resnext101_triplet_Adam_init_two_branch_distill_layer4_no_finetune_bz94_again/net_200.pth', -1)]

    model_base5 = [('/raid/home/zhihui/reid_strong_baseline/logs/resnet101_new_softmax_triplet_Adam_H384_W128/net_180.pth', -1)]
    #result = restore_file('features/market_two_branch_softmax_baseline_result_market_best.pkl')
    #names = result['names']
    #feats = result['features']
    #print(feats.shape)
    #print(len(names))
    #np.save('best_feats.npy', np.array(feats))
    #np.save('best_names.npy', np.array(names))
    #exit(0)

    for item in model_base5:
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
        #save_file(result, 'features/market_two_branch_softmax_baseline_result_market_best.pkl')
        txt_name = 'features/market_two_branch_softmax_baseline_result_market_best.txt'
        #txt_name = 'features/result_three_part_%s_softmax_H224_W224.txt' % str(part)
        #with open(txt_name, 'w') as f:
        #    f.write(ss)

        torch.cuda.empty_cache()


