# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os.path as osp
import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from torch.autograd import Variable
from utils.reid_metric import R1_mAP#, R1_mAP_reranking
from engine.inference import create_supervised_evaluator

from myLoss import LSR, LSR_direction, AdaptiveLSR

import functions


global ITER, best_mAP, best_epoch, mAP_path, save_flag, model_dir, freeze_flag
ITER = 0
best_mAP = -1
best_epoch = -1
mAP_path = ''
save_flag = False
model_dir = ''
freeze_flag = False


def save_network(network, path):
    file_path = os.path.join(path)
    print('*** Saving ', file_path)
    if isinstance(network, torch.nn.DataParallel):
        torch.save(network.module.state_dict(), file_path)
    else:
        torch.save(network.state_dict(), file_path)


def mixup_data(x, y, alpha=0.35, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, lam, y_a, y_b

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data2(x, y1, y2, alpha=0.35, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x, lam, y1_a, y1_b, y2_a, y2_b

# Random Image Cropping and Patching
def RICAP(inputs, target, alpha=0.35):
    #print('inputs.shape:', inputs.shape)
    I_x, I_y = inputs.size()[2:]
    w = int(np.round(I_x * np.random.beta(alpha, alpha)))
    h = int(np.round(I_y * np.random.beta(alpha, alpha)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        idx = torch.randperm(inputs.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = inputs[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = target[idx].cuda()
        W_[k] = w_[k] * h_[k] / (I_x * I_y)

    patched_images = torch.cat((torch.cat((cropped_images[0], cropped_images[1]), 2),
                                torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
    patched_images = patched_images.cuda()
    #print('patched_images.shape:', patched_images.shape)
    return patched_images, c_, W_

def remove_past_model(model_dir, best_name):
    path_list = os.listdir(model_dir)
    for path in path_list:
        if path[-4:] == '.pth' and path[:4] == 'net_':
            if path != best_name:
                os.remove(osp.join(model_dir, path))
                print('+++ delete ', path)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_supervised_trainer(model, optimizer, loss_fn, criterion, criterion_mse, criterion_lsr, criterion_adaptive_lsr, criterion_lsr_direction,
                              mask_tensor_matrix, device=None, mixup=False, ricap=False, freeze_layer=False, freeze_epoch=25):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    #global save_flag, model_dir

    print('\n\n\n+++++++++++++ Variable+++++++++\n\n\n')
    #mixup=False
    print('+++ use mixup: ', mixup, '  alpha:', 0.35)
    #ricap=False
    print('+++ use ricap: ', ricap, '  alpha:', 0.35)

    def _update(engine, batch):
        global save_flag, model_dir, best_epoch, freeze_flag
        if save_flag:
            save_flag = False
            print(' set save_flag: False')
            model_path = osp.join(model_dir, 'net_%s.pth' % str(best_epoch))
            save_network(model, model_path)
            remove_past_model(model_dir, 'net_%s.pth' % str(best_epoch))

        model.train()
        optimizer.zero_grad()
        img, target, viewids, new_labels = batch
        #print(target)
        #print('\n\n\n')
        #print(viewids)
        if freeze_layer and freeze_flag and (freeze_epoch == engine.state.epoch):
            #functions.freeze_layer(model, 'base', True)
            functions.freeze_global_model(model, True)
            freeze_flag = False

        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        #viewids = viewids.to(device) if torch.cuda.device_count() >= 1 else viewids
        new_labels = new_labels.to(device) if torch.cuda.device_count() >= 1 else new_labels

        if mixup:
            #inputs, targets_a, targets_b, lam = mixup_data(img, target)
            inputs, lam, target_a, target_b, new_labels_a, new_labels_b  = mixup_data2(img, target, new_labels)
            inputs, target_a, target_b, new_labels_a, new_labels_b = Variable(inputs), Variable(target_a), Variable(target_b), \
                                                                     Variable(new_labels_a), Variable(new_labels_b)
            score, score2, feat = model(img, target)
            loss1_func = mixup_criterion(target_a, target_b, lam)
            loss2_func = mixup_criterion(new_labels_a, new_labels_b, lam)
            loss1 = torch.sum(loss1_func(criterion_lsr, score))
            loss2 = torch.sum(loss2_func(criterion_lsr_direction, score2))
        elif ricap:
            patched_images, c_, W_ = RICAP(img, target, alpha=0.35)
            img, target, new_labels = Variable(img), Variable(target), Variable(new_labels)
            score, score2, feat = model(patched_images, target)
            loss1 = sum([W_[k] * criterion_lsr(score, c_[k]) for k in range(4)])
            #acc = sum([W_[k] * accuracy(score, c_[k])[0] for k in range(4)])
            #print('  acc:', acc.item())
            loss2 = torch.sum(criterion_lsr_direction(score2, new_labels))
        else:
            img, target, new_labels = Variable(img), Variable(target), Variable(new_labels)
            #viewids = Variable(viewids)
#            print(' Here 1 ........')
            #score, score2, feat, feat1, feat2, feat1_distill, feat2_distill = model(img, target)
            score, score2, feat = model(img, target)
            #if engine.state.epoch >= 70:
            #    score, score2, feat = model(img, target, True)
            #    loss1 = torch.mean(score)
            #else:
            #    score, score2, feat = model(img, target)
            #    loss1 = torch.sum(criterion_lsr(score, target))
            #print(score.shape)
#            print(' Here 2 ++++++++++++++++++++')
            #score, feat = model(img, target)
            #print(' Here 2 ++++++++++++++++++++')
            # lsr
#            loss1 = torch.mean(score)
#            loss1 = torch.sum(criterion_lsr(score, target))
            #loss1 = torch.sum(criterion_adaptive_lsr(score, target))
            #loss_v = torch.sum(criterion_lsr(score3, viewids))
            loss = loss_fn(score, feat, target)
            #print(' Here 3 ++++++++++++++++++++')
            #loss = loss1+ loss2
            #loss2 = torch.sum(criterion_lsr(score2, target))
            #loss2 = 0
            #if engine.state.epoch <= 15:
            #    loss = loss1
            #else:
            #    loss = loss1 + loss2
            #loss_2_1 = torch.sum(criterion_lsr(score_2, target))
            # lsr_direction
            #loss2 = 0
            #loss2 = torch.sum(criterion_lsr_direction(score2, new_labels))

            #feat1_distill = torch.nn.functional.normalize(feat1_distill, dim=1, p=2)
            #feat2_distill = torch.nn.functional.normalize(feat2_distill, dim=1, p=2)
            #loss4 = torch.sum(criterion_mse(feat1_distill, feat2_distill))
            #loss4 = 0
            #show_loss2 = loss2.item()
            #loss2 = torch.sum(loss_fn(score2, feat, new_labels))
        # fc loss
        loss3 = 0
        show_loss3 = 0
        if engine.state.epoch >= 500:
            if isinstance(model, torch.nn.DataParallel):
                fc_weight = list(model.module.classifier2.parameters())[0]  # [12855, 768]
            else:
                fc_weight = list(model.classifier2.parameters())[0]

            #print('type fc_weight:', type(fc_weight))
            n = fc_weight.size(0)
            euc_dist = torch.pow(fc_weight, 2).sum(dim=1, keepdim=True).expand(n, n)
            euc_dist = euc_dist + euc_dist.t()
            euc_dist.addmm_(1, -2, fc_weight, fc_weight.t())
            euc_dist = euc_dist.clamp(min=1e-12).sqrt()  # for numerical stability
            #print('type euc_dist:', type(euc_dist))
            euc_dist = euc_dist * mask_tensor_matrix
            #print('type euc_dist:', type(euc_dist))
            loss3 = torch.sum(euc_dist)
            show_loss3 = loss3.item()

        loss_2_4 = 0
        show_loss_2_4 = 0
        if engine.state.epoch  >= 500:
            loss_2_4 = loss_fn(score, feat1, target) + loss_fn(score, feat2, target)
            show_loss_2_4 = loss_2_4.item()

        alpha = 0.1
        beta = 1
        gama = 1
        if engine.state.epoch >200:
            gama = 0.3
        if engine.state.epoch >= 500:
            beta = 1
        #loss = loss1 + (loss2*gama + loss3 * alpha) #+ loss_2_4 + loss4*0.5 + loss_v * 0.5
        #print(loss)
        #loss_2 = loss_2_1 + loss_2_4
        #loss = loss_fn(score, feat, target)
        #loss = loss_1 + loss_2
#        print(' backward')
        loss.backward()
        #loss_1.backward(retain_graph=True)
        #loss_2.backward()

        #score, feat = model(img)
        #loss = torch.sum(loss_fn(score, feat, target)) # * 10
        #loss2 = torch.sum(loss_fn(score2, feat, viewids))
        #loss = loss1 + loss2
        #loss.backward()
        optimizer.step()
        #print('   here-4')
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        #acc = torch.tensor(0)
        #acc2 = (score2.max(1)[1] == new_labels).float().mean()
        #acc3 = (score3.max(1)[1] == viewids).float().mean()
        #print('    epoch:', engine.state.epoch, 'loss:', round(loss.item(),5),'loss1:', round(loss1.item(),5), 'loss2:', round(loss2.item(),5), '  stripes acc:', acc.item(), 'global_acc:', acc2.item())
        print('    epoch:', engine.state.epoch, 'loss:', round(loss.item(),5), '  avg acc:', acc.item())#, '  max acc:', acc2.item(), '  viewid acc:', acc3.item())
        #acc_2 = (score_2.max(1)[1] == target).float().mean()
        #print('    epoch:', engine.state.epoch, 'loss:', round(loss.item(),5),'loss_1:', round(loss_1.item(),5), 'loss_2:', round(loss_2.item(),5),
        #      'loss1:', round(loss1.item(),5), 'loss2:', round(show_loss2,5), 'loss3:', round(show_loss3,5), 'loss_2_1:',round(loss_2_1.item(),5),
        #      'loss_2_4:', round(show_loss_2_4, 8), 'alpha:', alpha, 'beta:', beta, 'gama:', gama,'acc:', acc.item(), 'acc2:', acc2.item(), 'acc_2:', acc_2.item())
        #print('    epoch:', engine.state.epoch, 'loss:', round(loss.item(),5), 'acc:', acc.item())
        return loss.item(), acc.item()

    return Engine(_update)


#def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
#                              device=None):
#    """
#    Factory function for creating a trainer for supervised models
#
#    Args:
#        model (`torch.nn.Module`): the model to train
#        optimizer (`torch.optim.Optimizer`): the optimizer to use
#        loss_fn (torch.nn loss function): the loss function to use
#        device (str, optional): device type specification (default: None).
#            Applies to both model and batches.
#
#    Returns:
#        Engine: a trainer engine with supervised update function
#    """
#    if device:
#        if torch.cuda.device_count() > 1:
#            model = nn.DataParallel(model)
#        model.to(device)
#
#    def _update(engine, batch):
#
#        model.train()
#        optimizer.zero_grad()
#        optimizer_center.zero_grad()
#        img, target, viewids = batch
#        print(target)
#        print(viewids)
#        img = img.to(device) if torch.cuda.device_count() >= 1 else img
#        target = target.to(device) if torch.cuda.device_count() >= 1 else target
#        viewids = viewids.to(device) if torch.cuda.device_count() >= 1 else viewids
#        score, score2, feat = model(img)
#        loss1 = loss_fn(score, feat, target)
#        loss2 = loss_fn(score2, feat, viewids)
#        loss = loss1 + loss2
#
#        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
#        loss.backward()
#        optimizer.step()
#        for param in center_criterion.parameters():
#            param.grad.data *= (1. / cetner_loss_weight)
#        optimizer_center.step()
#
#        # compute acc
#        acc = (score.max(1)[1] == target).float().mean()
#        print('loss:', loss.item(),'loss1:', loss1.item(),'loss2:', loss2.item())
#        return loss.item(), acc.item()
#
#    return Engine(_update)


#def create_supervised_evaluator(model, metrics,
#                                device=None):
#    """
#    Factory function for creating an evaluator for supervised models
#
#    Args:
#        model (`torch.nn.Module`): the model to train
#        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
#        device (str, optional): device type specification (default: None).
#            Applies to both model and batches.
#    Returns:
#        Engine: an evaluator engine with supervised inference function
#    """
#    if device:
#        if torch.cuda.device_count() > 1:
#            model = nn.DataParallel(model)
#        model.to(device)
#
#    def _inference(engine, batch):
#        model.eval()
#        with torch.no_grad():
#            data, pids, camids = batch
#            data = data.to(device) if torch.cuda.device_count() >= 1 else data
#            feat = model(data)
#            return feat, pids, camids
#
#    engine = Engine(_inference)
#
#    for name, metric in metrics.items():
#        metric.attach(engine, name)
#
#    return engine


def get_same_id_list(image_map_label2):
    print('--get_same_id_list')
    all_id_list = [-1]*len(image_map_label2)
    for i in image_map_label2.keys():
        all_id_list[image_map_label2[i]] = int(i[0])

    id_list = list(set(all_id_list))
    all_id_list = np.array(all_id_list)
    same_id_list = []
    for i in id_list:
        index_list = np.where(all_id_list==i)
        same_id_list.append(list(index_list[0]))
    print('len(same_id_list):', len(same_id_list))
    print('same_id_list:', same_id_list[:20])

    return same_id_list


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        image_map_label2,
        num_classes2):

    # ---------------------- LOSS start-----------------------------
    print('----------Initialize Loss Start...')
    criterion = torch.nn.CrossEntropyLoss()
    criterion_lsr = LSR()
    criterion_mse = torch.nn.MSELoss() #(size_average=True)
    criterion_lsr_direction = LSR_direction()
    criterion_adaptive_lsr = AdaptiveLSR(0.25)

    criterion_lsr.set_epsilon(0.1)

    criterion_lsr_direction.set_alpha(0.6)
    criterion_lsr_direction.set_beta(0.15)
    print('******\nalpha:', criterion_lsr_direction.alpha, ' beta:', criterion_lsr_direction.beta)
    same_id_list = get_same_id_list(image_map_label2)
    criterion_lsr_direction.set_mask(same_id_list, num_classes2)

    mask_tensor_matrix = torch.zeros(num_classes2, num_classes2)
    eplsion=[1,1,1]
    for ids_item in same_id_list:
        if len(ids_item) == 2:
            mask_tensor_matrix[ids_item[0], ids_item[1]] = eplsion[1]
        if len(ids_item) == 3:
            mask_tensor_matrix[ids_item[0], ids_item[1]] = eplsion[2]/3
            mask_tensor_matrix[ids_item[0], ids_item[2]] = eplsion[2]/3
            mask_tensor_matrix[ids_item[1], ids_item[2]] = eplsion[2]/3
    mask_tensor_matrix = mask_tensor_matrix.float()
    #mask_tensor_matrix = Variable(mask_tensor_matrix.cuda())
    print('mask_tensor_matrix.shape:', mask_tensor_matrix.shape, type(mask_tensor_matrix), '\n\n\n')
    print('----------Initialize Loss End!!!')
    # ---------------------------------------------------------

    global mAP_path, model_dir
    mAP_path = osp.join(cfg.OUTPUT_DIR, 'map_cmc.txt')
    model_dir = cfg.OUTPUT_DIR

    map_cmc_txt = open(mAP_path, 'a+')
    map_cmc_txt.close()

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, criterion, criterion_mse, criterion_lsr, criterion_adaptive_lsr, criterion_lsr_direction, mask_tensor_matrix,
                                        device, cfg.SOLVER.MIXUP, cfg.SOLVER.RICAP, cfg.MODEL.FREEZE_BASE, cfg.MODEL.FREEZE_BASE_EPOCH)
    #evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=3, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, #.state_dict(),
                                                                     'optimizer': optimizer})#.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        if cfg.SOLVER.MY_WARMUP == 'yes':
            if engine.state.epoch <= cfg.SOLVER.MY_WARMUP_EPOCH:
                print('--- warmup')
            else:
                scheduler.step()
        else:
            scheduler.step()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            if cfg.SOLVER.MY_SCHEDULER == 'yes':
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}"
                            .format(engine.state.epoch, ITER, len(train_loader),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc']))
            else:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(engine.state.epoch, ITER, len(train_loader),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                    scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.2f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        global best_mAP, best_epoch, mAP_path, save_flag
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("[Epoch {}]  mAP: {:.2%}".format(engine.state.epoch, mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
            if float(mAP) > float(best_mAP):
                print('+++ get best_mAP: ', best_mAP, '-->' ,mAP)
                best_mAP = mAP
                best_epoch = int(engine.state.epoch)
                save_flag = True
                print(' set save_flag: True')

            map_cmc_txt = open(mAP_path, 'a+')
            map_cmc_txt.write("Epoch[{}]    best_mAP: {:.2f}  best_epoch: {} \n".format(
                               engine.state.epoch, best_mAP*100, best_epoch))
            map_cmc_txt.write("       mAP: {:.2f}  Rank-1: {:.2f}  Rank-5: {:.2f}  Rank-10: {:.2f}  Rank-20: {:.2f}\n".format(
                               float(mAP)*100, cmc[0]*100, cmc[4]*100, cmc[9]*100, cmc[19]*100))
            map_cmc_txt.flush()
            os.fsync(map_cmc_txt)
            map_cmc_txt.close()


    trainer.run(train_loader, max_epochs=epochs)


#def do_train_with_center(
#        cfg,
#        model,
#        center_criterion,
#        train_loader,
#        val_loader,
#        optimizer,
#        optimizer_center,
#        scheduler,
#        loss_fn,
#        num_query,
#        start_epoch
#):
#    log_period = cfg.SOLVER.LOG_PERIOD
#    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
#    eval_period = cfg.SOLVER.EVAL_PERIOD
#    output_dir = cfg.OUTPUT_DIR
#    device = cfg.MODEL.DEVICE
#    epochs = cfg.SOLVER.MAX_EPOCHS
#
#    logger = logging.getLogger("reid_baseline.train")
#    logger.info("Start training")
#    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
#    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
#    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
#    timer = Timer(average=True)
#
#    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
#                                                                     'optimizer': optimizer.state_dict(),
#                                                                     'optimizer_center': optimizer_center.state_dict()})
#
#    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
#                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
#
#    # average metric to attach on trainer
#    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
#    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
#
#    @trainer.on(Events.STARTED)
#    def start_training(engine):
#        engine.state.epoch = start_epoch
#
#    @trainer.on(Events.EPOCH_STARTED)
#    def adjust_learning_rate(engine):
#        scheduler.step()
#
#    @trainer.on(Events.ITERATION_COMPLETED)
#    def log_training_loss(engine):
#        global ITER
#        ITER += 1
#
#        if ITER % log_period == 0:
#            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
#                        .format(engine.state.epoch, ITER, len(train_loader),
#                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
#                                scheduler.get_lr()[0]))
#        if len(train_loader) == ITER:
#            ITER = 0
#
#    # adding handlers using `trainer.on` decorator API
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def print_times(engine):
#        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
#                    .format(engine.state.epoch, timer.value() * timer.step_count,
#                            train_loader.batch_size / timer.value()))
#        logger.info('-' * 10)
#        timer.reset()
#
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_validation_results(engine):
#        if engine.state.epoch % eval_period == 0:
#            evaluator.run(val_loader)
#            cmc, mAP = evaluator.state.metrics['r1_mAP']
#            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
#            logger.info("mAP: {:.1%}".format(mAP))
#            for r in [1, 5, 10]:
#                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#
#    trainer.run(train_loader, max_epochs=epochs)
