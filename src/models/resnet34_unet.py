# ResNet-34 作為 Encoder，並結合 UNet 的 Decoder

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet34_UNet(nn.Module):
    #pretrained預訓練
    def __init__(self, num_classes=1, pretrained=False):
        super(ResNet34_UNet, self).__init__()

        #載入ResNet34作為Encoder
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        self.encoder = nn.Sequential(*list(resnet34.children())[:-2])  # 移除最後的全連接層與池化層

        #做encoder
        #使用ResNet34函式
        #Layer 0: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #Layer 1: BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #Layer 2: ReLU(inplace=True)
        self.enc1 = nn.Sequential(*list(resnet34.children())[:3])  # 第一層 (64 channels)
        self.enc2 = resnet34.layer1  # 輸出: 64 x 64 x 64
        self.enc3 = resnet34.layer2  # 輸出: 128 x 32 x 32
        self.enc4 = resnet34.layer3  # 輸出: 256 x 16 x 16
        self.enc5 = resnet34.layer4  # 輸出: 512 x 8 x 8

        #decoder
        self.dec5 = self._decoder_block(512,256)
        self.dec4 = self._decoder_block(256+256, 128)
        self.dec3 = self._decoder_block(128+128, 64)
        self.dec2 = self._decoder_block(64+64, 64)
        self.dec1 = self._decoder_block(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1) # 1x1 卷積，輸出 mask



    def _decoder_block(self, in_channels, out_channels):
        """
        建立decoder，使轉置捲積向上採樣
        nn.ConvTranspose2d enc5 (512, 8, 8) 變成 dec5 (256, 16, 16) 的關鍵
        """
        return nn.Sequential(
            #stride步長為 2
            #padding在輸入周圍填充 1 像素，確保輸出大小合適
            #output_padding=1 可以幫助補回這個丟失的像素
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            #轉置卷積（ConvTranspose2d）：上採樣（Upsampling）
            #普通卷積（Conv2d）：進一步特徵學習
            #ReLU 激活函數（ReLU(inplace=True)）：引入非線性，使模型更具表現力
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #inplace=True是否在原地進行運算，非線性能力
            nn.ReLU(inplace=True)

        )
    
    def forward(self, x):
        #encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        #decoder
        #UNet 架構 Skip Connection (跳層連接)
        #enc4 來自 Encoder，它包含 較高解析度的特徵，但 深度較淺 (早期的特徵)。
        #dec5 來自 Decoder，它包含 更高層次的語意資訊，但 解析度較低 (深層特徵)。
        #enc1.shape[2:]取 enc1 的高度 (H) 和寬度 (W) 
        #mode="bilinear"使用 雙線性插值 來進行上採樣
        #align_corners=False	避免像素偏移問題，讓插值結果更加平滑
        dec5 = self.dec5(enc5)
        dec5 = F.interpolate(dec5, size=enc4.shape[2:], mode="bilinear", align_corners=False)

        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec4 = F.interpolate(dec4, size=enc3.shape[2:], mode="bilinear", align_corners=False)

        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode="bilinear", align_corners=False)

        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode="bilinear", align_corners=False)

        dec1 = self.dec1(dec2)

        #final output
        #1x1 卷積不會影響空間資訊，只會影響通道數
        out = self.final_conv(dec1) # num_classes x 256 x 256

        return out






