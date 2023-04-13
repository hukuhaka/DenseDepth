import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, pred, gt_depth, mask=None, interpolate=True):

        if mask is not None:
            pred = torch.clip(pred[mask], min=1e-06)
            gt_depth = torch.clip(gt_depth[mask], min=1e-06)
        else:
            pred = torch.clip(pred, min=1e-06)
            gt_depth = torch.clip(gt_depth, min=1e-06)

        g = torch.log(pred) - torch.log(gt_depth)
        Dg = torch.mean(torch.pow(g, 2)) - 0.85 * torch.pow(torch.mean(g), 2)

        return 10 * torch.sqrt(Dg)
