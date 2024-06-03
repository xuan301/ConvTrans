import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfFocalLoss(nn.CrossEntropyLoss):
    def forward(self, inp, target):
        ce_loss = F.cross_entropy(inp, target.long(), weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = 1 * (1 - pt) ** 2 * ce_loss
        return focal_loss

def get_loss_module():
    # return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample
    return SelfFocalLoss()  # outputs loss for each batch sample

def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
