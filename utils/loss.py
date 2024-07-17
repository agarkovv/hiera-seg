import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class FocalTreeMinLoss(nn.Module):
    def __init__(self):
        super(FocalTreeMinLoss, self).__init__()

    def forward(self, inputs, targets):
        raise NotImplementedError


class TreeTripletLoss(nn.Module):
    def __init__(self):
        super(TreeTripletLoss, self).__init__()

    def forward(self, inputs, targets):
        raise NotImplementedError


class HierarchicalLoss(nn.Module):
    def __init__(self, beta=0.5, schedule_beta=False):
        super(HierarchicalLoss, self).__init__()
        self.beta = beta
        self.schedule_beta = schedule_beta
        if schedule_beta:
            raise NotImplementedError

        self.ftm_loss = FocalTreeMinLoss()
        self.tt_loss = TreeTripletLoss()

    def forward(self, inputs, targets):
        raise NotImplementedError
