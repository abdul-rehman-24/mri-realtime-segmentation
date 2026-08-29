
import albumentations as A
import cv2

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR,
                  mask_interpolation=cv2.INTER_NEAREST,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
        A.ElasticTransform(alpha=15, sigma=3, interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    ])

def get_val_transform():
    return A.Compose([])
