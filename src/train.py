
import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    from metrics.dice_metric import compute_dice_score
    model.train()
    total_loss, total_dice = 0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += compute_dice_score(logits, masks)
    return total_loss / len(loader), total_dice / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    from metrics.dice_metric import compute_dice_score
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_dice += compute_dice_score(logits, masks)
    return total_loss / len(loader), total_dice / len(loader)
