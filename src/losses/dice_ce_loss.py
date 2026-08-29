
import torch
import torch.nn as nn


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=5, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_per_class.mean()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice
