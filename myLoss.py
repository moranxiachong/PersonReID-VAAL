from __future__ import absolute_import
import sys

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import random
import math
import torch.nn.functional as F

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss', 'MSMLLoss']

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss



class LSR(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LSR, self).__init__()
        self.epsilon = epsilon
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, inputs, targets):
        num_class = inputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        #print (targets, 'lsr')
        outputs = self.log_softmax(inputs)
        loss = - (targets * outputs)
        #print (loss.size(), 'lsr')
        loss = loss.sum(dim=1)
        #print (loss, 'lsr')
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 1 - self.epsilon)
        targets_onehot.add_(self.epsilon / (num_class - 1))
        return targets_onehot

    def set_epsilon(self, option):
        self.epsilon=option
        print('LSR new epsilon:', self.epsilon)

class JointLoss(nn.Module):
    """docstring for ClassName"""
    def __init__(self, alpha=1):
        super(JointLoss, self).__init__()
        self.mask = None
        self.alpha = alpha
        self.log_softmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.Softmax()

    def forward(self, inputs, targets):
        num_class = inputs.size()[1]
        inputs = self.softmax(inputs)

        targets = self.mask[targets.data.cpu()]
        targets = Variable(targets.cuda())
        
        results = targets * inputs
        results = results.sum(dim=1)
        
        loss = torch.log(results)
        loss = loss.mean(dim=0) * -1
        return loss

    def set_mask(self, same_id_list, num_class):
        mask = torch.FloatTensor(num_class, num_class)
        mask.zero_()
        tmp = torch.LongTensor(list(range(num_class)))
        tmp = torch.unsqueeze(tmp, 1)
        mask.scatter_(1, tmp, self.alpha)
        for id_list in same_id_list:
            if len(id_list) == 2:
                mask[id_list[0], id_list[1]] = self.alpha
                mask[id_list[1], id_list[0]] = self.alpha
            if len(id_list) == 3:
                mask[id_list[0], id_list[1]] = self.alpha
                mask[id_list[0], id_list[2]] = self.alpha
                mask[id_list[1], id_list[0]] = self.alpha
                mask[id_list[1], id_list[2]] = self.alpha
                mask[id_list[2], id_list[0]] = self.alpha
                mask[id_list[2], id_list[1]] = self.alpha
        self.mask = mask

class LSR_block(nn.Module):
    def __init__(self, alpha=0.2, beta=0.1, topk=30):
        super(LSR_block, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.mask = None
        self.mask2 = None
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, inputs, targets):  # torch.Size([256, 10431])  torch.Size([256])
        num_class = inputs.size()[1]

        filter_mask = self.mask2[targets.data.cpu()]
        targets = self.mask[targets.data.cpu()]

        targets = Variable(targets.cuda())
        filter_mask = Variable(filter_mask.cuda())

        outputs = self.log_softmax(inputs*filter_mask)

        loss = - (targets * outputs)
        #print (loss.size(), 'lsr')
        loss = loss.sum(dim=1)
        #print (loss, 'lsr')
        loss = loss.mean(dim=0)
        return loss

    def set_mask(self, same_id_list, num_class):
        print('self.alpha:', self.alpha, 'self.beta:', self.beta)
        mask = torch.FloatTensor(num_class, num_class)
        mask2 = torch.FloatTensor(num_class, num_class)

        mask.zero_()
        mask2.zero_()
        tmp = torch.LongTensor(list(range(num_class)))
        tmp = torch.unsqueeze(tmp, 1)
        mask.scatter_(1, tmp, self.alpha)
        for i in range(num_class):
            for j in range(num_class):
                mask2[i][j] = 1.0

        for id_list in same_id_list:
            if len(id_list) == 2:
                mask[id_list[0], id_list[1]] = self.beta
                mask[id_list[1], id_list[0]] = self.beta
                mask[id_list[0]].add_((1-self.alpha-self.beta) / num_class)
                mask[id_list[1]].add_((1-self.alpha-self.beta) / num_class)

                mask2[id_list[0], id_list[1]] = 0.0
                mask2[id_list[1], id_list[0]] = 0.0

            if len(id_list) == 3:
                mask[id_list[0], id_list[1]] = self.beta
                mask[id_list[0], id_list[2]] = self.beta
                mask[id_list[1], id_list[0]] = self.beta
                mask[id_list[1], id_list[2]] = self.beta
                mask[id_list[2], id_list[0]] = self.beta
                mask[id_list[2], id_list[1]] = self.beta

                mask[id_list[0]].add_((1-self.alpha-2*self.beta) / num_class)
                mask[id_list[1]].add_((1-self.alpha-2*self.beta) / num_class)
                mask[id_list[2]].add_((1-self.alpha-2*self.beta) / num_class)

                mask2[id_list[0], id_list[1]] = 0.0
                mask2[id_list[0], id_list[2]] = 0.0
                mask2[id_list[1], id_list[0]] = 0.0
                mask2[id_list[1], id_list[2]] = 0.0
                mask2[id_list[2], id_list[0]] = 0.0
                mask2[id_list[2], id_list[1]] = 0.0

        self.mask = mask
        self.mask2 = mask2

    def set_alpha(self, option):
        self.alpha = option

    def set_beta(self, option):
        self.beta = option

class LSR_direction(nn.Module):
    """docstring for ClassName"""
    def __init__(self, alpha=0.2, beta=0.1, topk=30):
        super(LSR_direction, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.mask = None
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, inputs, targets, which_mask='mask'):  # torch.Size([256, 10431])  torch.Size([256])
        num_class = inputs.size()[1]
        if which_mask=='mask':
            targets = self.mask[targets.data.cpu()]
        else:
            targets = self.mask2[targets.data.cpu()]
        targets = Variable(targets.cuda())

        outputs = self.log_softmax(inputs)
        loss = - (targets * outputs)
        #print (loss.size(), 'lsr')
        loss = loss.sum(dim=1)
        #print (loss, 'lsr')
        loss = loss.mean(dim=0)
        return loss

    def set_mask(self, same_id_list, num_class):
        print('self.alpha:', self.alpha, 'self.beta:', self.beta)
        mask = torch.FloatTensor(num_class, num_class)
        mask.zero_()
        tmp = torch.LongTensor(list(range(num_class)))
        tmp = torch.unsqueeze(tmp, 1)
        mask.scatter_(1, tmp, self.alpha)
        for id_list in same_id_list:
            if len(id_list) == 2:
                mask[id_list[0], id_list[1]] = self.beta*2
                mask[id_list[1], id_list[0]] = self.beta*2
                mask[id_list[0]].add_((1-self.alpha-self.beta*2) / num_class)
                mask[id_list[1]].add_((1-self.alpha-self.beta*2) / num_class)
            if len(id_list) == 3:
                mask[id_list[0], id_list[1]] = self.beta
                mask[id_list[0], id_list[2]] = self.beta
                mask[id_list[1], id_list[0]] = self.beta
                mask[id_list[1], id_list[2]] = self.beta
                mask[id_list[2], id_list[0]] = self.beta
                mask[id_list[2], id_list[1]] = self.beta

                mask[id_list[0]].add_((1-self.alpha-2*self.beta) / num_class)
                mask[id_list[1]].add_((1-self.alpha-2*self.beta) / num_class)
                mask[id_list[2]].add_((1-self.alpha-2*self.beta) / num_class)

        self.mask = mask
    
    def set_alpha(self, option):
        self.alpha = option

    def set_beta(self, option):
        self.beta = option

class LSR_topk(nn.Module):
    def __init__(self, epsilon=0.1, topk=39935):
        super(LSR_topk, self).__init__()
        self.epsilon = epsilon
        self.log_softmax = torch.nn.LogSoftmax()
        self.topk = topk

    def forward(self, inputs, targets):
        onehot = self._get_one_hot(inputs, targets.data.cpu())
        targets = Variable(onehot.cuda())
        #print (targets, 'topk')
        outputs = self.log_softmax(inputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _get_one_hot(self, inputs,  targets):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], inputs.size()[1])
        targets_onehot.zero_()
        
        topk_idx = torch.topk(inputs, self.topk, 1)[1].detach()

        targets_onehot.scatter_(1, topk_idx.cpu().data, self.epsilon / self.topk)
        targets_onehot.scatter_(1, targets, 1 - self.epsilon)
        

        #targets_onehot.add_(self.epsilon / topk)
        return targets_onehot


class AdaptiveLSR(nn.Module):
    def __init__(self, epsilon=0.1):
        super(AdaptiveLSR, self).__init__()
        self.epsilon = epsilon
        self.log_softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, inputs, targets):
        num_class = inputs.size()[1]
        outputs = self.log_softmax(inputs)
        pt = Variable(outputs.data.exp())
        targets = self._class_to_one_hot(targets.data.cpu(), pt,  num_class)
        #targets2 = self._class_to_one_hot_fix(targets.data.cpu(), pt,  num_class)
  
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, pt, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        #print targets_onehot, targets_onehot.size()
        targets_onehot.scatter_(1, targets, 1)
        targets_onehot = Variable(targets_onehot.cuda())
        lsr_p = self.epsilon * (1 - pt)
        targets_onehot = targets_onehot * (1 - lsr_p) + lsr_p / (num_class - 1)
        
        #targets_onehot.scatter_(1, targets, 1 - pt.data.cpu() * self.epsilon)
        return targets_onehot
 
    def _class_to_one_hot_fix(self, targets, pt, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, (1 - self.epsilon * pt.data.cpu()))
        targets_onehot.add_(self.epsilon * pt.data.cpu() / (num_class - 1))
        
        return targets_onehot
    
   
if __name__ == '__main__':
    pass

