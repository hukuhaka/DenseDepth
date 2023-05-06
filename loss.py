import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, pred, gt_depth, mask=None, interpolate=True):

        if mask is not None:
            pred = pred[mask]
            gt_depth = gt_depth[mask]

        g = torch.log(pred) - torch.log(gt_depth)
        Dg = torch.mean(torch.pow(g, 2)) - 0.85 * torch.pow(torch.mean(g), 2)

        return 10 * torch.sqrt(Dg)
