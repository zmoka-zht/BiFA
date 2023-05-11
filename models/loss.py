import torch
import torch.nn as nn
from torch import Tensor, einsum
import torch.nn .functional as F
from misc.torchutils import class2one_hot,simplex
from models.darnet_help.loss_help import FocalLoss, dernet_dice_loss

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def dice_loss(predicts,target,weight=None):
    idc= [0, 1]
    probs = torch.softmax(predicts, dim=1)
    # target = target.unsqueeze(1)
    target = class2one_hot(target, 2)
    assert simplex(probs) and simplex(target)

    pc = probs[:, idc, ...].type(torch.float32)
    tc = target[:, idc, ...].type(torch.float32)
    intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

    divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

    loss = divided.mean()
    return loss

def ce_dice(input, target, weight=None):
    ce_loss = cross_entropy(input, target)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def hybrid_loss(predictions, target, weight=[0,2,0.2,0.2,0.2,0.2]):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i,prediction in enumerate(predictions):

        bce = cross_entropy(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i]*(bce + dice) #- ssimloss

    return loss

class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss