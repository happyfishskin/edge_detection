import torch
import torch.nn as nn
#提供激活函數、損失函數
import torch.nn.functional as F


class DoubleConv(nn.Module):
    #UNet的基本單元，包含兩個 3×3 卷積層
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #3x3捲積
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #加速收斂並穩定訓練
            nn.BatchNorm2d(out_channels),
            #加入非線性特性
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)   

class UNet(nn.Module):
    """
    UNet主結構
    包含Encoder和Decoder
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        #Encoder
        #4個DoubleConv層
        #每層的輸出通道數為 64 → 128 → 256 → 512
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        #下採樣，每次將特徵圖縮小一半（H/2, W/2）
        self.pool = nn.MaxPool2d(2)

        #Bottleneck橋階層
        #UNet最深的一層
        #增加通道數，並捕捉更高階的抽象特徵
        self.bottleneck = DoubleConv(512, 1024)

        #Decoder
        #上採樣，使特徵圖回到與輸入相同的大小

        #ConvTranspose2d反卷積，讓圖片放大2倍
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256,128 , kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        #Output layer
        #最後一層 1×1 卷積
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        #Encoder
        #進行下採樣，保留編碼特徵
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        #Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        #Decoder
        #ConvTranspose2d 上採樣
        dec4 = self.up4(bottleneck)
        #將 Encoder 層的輸出與 Decoder 層的輸入拼接。
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        #使用 sigmoid 讓輸出值範圍在 0~1 之間
        return torch.sigmoid(self.final_conv(dec1))

