# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

import torch
import pprint
class JointsMSELossNoReduction(nn.Module):
    def __init__(self, use_target_weight, logger):
        super(JointsMSELossNoReduction, self).__init__()
        self.criterion = lambda x,y: ((x-y)**2).sum(1).unsqueeze(1)
        self.use_target_weight = use_target_weight
        self.logger= logger

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # N x J x 256 * 256
        # N x J
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = []

        for idx in range(num_joints):
            # N x (256*256)
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            #self.logger.info("Heatmap Shape Start" + str(heatmap_pred.shape))
            if self.use_target_weight:
                heatmap_pred = heatmap_pred.mul(target_weight[:, idx])
                #self.logger.info("Heatmap Shape Masked" + str(heatmap_pred.shape))
                loss_val = self.criterion(
                    heatmap_pred,
                    heatmap_gt.mul(target_weight[:, idx]))
                loss.append(0.5 * loss_val)
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))
        
        loss = torch.cat(loss,dim=1)
        #self.logger.info("Loss shape" + str(loss.shape))
        #pprint.pformat(loss.shape)
        return loss
