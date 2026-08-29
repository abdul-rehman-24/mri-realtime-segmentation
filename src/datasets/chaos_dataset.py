
import os
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset

ORGAN_VALUES = {63: 1, 126: 2, 189: 3, 252: 4}
CLASS_NAMES = {0: "Background", 1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}


class CHAOSDataset(Dataset):
    def __init__(self, patient_ids, mr_path, transform=None, target_size=256):
        import sys
        sys.path.append('/content/drive/MyDrive/mri-realtime-segmentation/src')
        from preprocessing.transforms import normalize_mri, resize_mri, resize_mask
        self._normalize_mri = normalize_mri
        self._resize_mri = resize_mri
        self._resize_mask = resize_mask

        self.samples = []
        self.transform = transform
        self.target_size = target_size

        for pid in patient_ids:
            t1dual = os.path.join(mr_path, pid, "T1DUAL")
            dicom_dir = os.path.join(t1dual, "DICOM_anon", "InPhase")
            ground_dir = os.path.join(t1dual, "Ground")
            dicom_files = sorted(os.listdir(dicom_dir))
            mask_files = sorted(os.listdir(ground_dir))
            for df, mf in zip(dicom_files, mask_files):
                self.samples.append((os.path.join(dicom_dir, df), os.path.join(ground_dir, mf)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dicom_path, mask_path = self.samples[idx]
        ds = pydicom.dcmread(dicom_path)
        mri = ds.pixel_array.astype(np.float32)

        raw_mask = np.array(Image.open(mask_path))
        mask = np.zeros_like(raw_mask, dtype=np.uint8)
        for val, class_idx in ORGAN_VALUES.items():
            region = (raw_mask >= val - 7) & (raw_mask <= val + 7)
            mask[region] = class_idx

        mri = self._normalize_mri(mri)
        mri = self._resize_mri(mri, self.target_size)
        mask = self._resize_mask(mask, self.target_size)

        if self.transform is not None:
            augmented = self.transform(image=mri, mask=mask)
            mri, mask = augmented['image'], augmented['mask']

        mri_tensor = torch.from_numpy(mri).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()
        return mri_tensor, mask_tensor
