
import torch.nn as nn
import sys
sys.path.append('/content/drive/MyDrive/mri-realtime-segmentation/src')
from models.blocks import DoubleConv, Encoder, Decoder


class BaselineUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        self.enc1 = Encoder(in_channels, 32)
        self.enc2 = Encoder(32, 64)
        self.enc3 = Encoder(64, 128)
        self.enc4 = Encoder(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.dec4 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec2 = Decoder(128, 64)
        self.dec1 = Decoder(64, 32)
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.seg_head(x)
