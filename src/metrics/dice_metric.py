
import torch

def compute_dice_score(logits, targets, num_classes=5, smooth=1e-5):
    preds = torch.argmax(logits, dim=1)
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    intersection = (preds_onehot * targets_onehot).sum(dim=(2, 3))
    union = preds_onehot.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
    dice_per_class = (2 * intersection + smooth) / (union + smooth)
    return dice_per_class.mean(dim=1).mean().item()
