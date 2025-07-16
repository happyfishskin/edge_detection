# train.py：此檔包含用於訓練神經網路模型的代碼。它包括函數
#與模型訓練、優化和反向傳播相關。

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

# 修正 import，確保 `train.py` 能找到 `models` 和 `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 正確 import 模組
from models.unet import UNet  # 來自 src/models/unet.py
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset,OxfordPetDataset  # 來自 src/oxford_pet.py
from utils import dice_score, plot_dice_score # 來自 src/utils.py
from evaluate import evaluate

def dice_loss(pred_mask, gt_mask, smooth=1e-5):
    pred_mask = torch.sigmoid(pred_mask)  # 確保輸出在 (0,1) 之間
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    intersection = (pred_mask * gt_mask).sum(dim=(2,3))
    union = pred_mask.sum(dim=(2,3)) + gt_mask.sum(dim=(2,3))

    loss = 1 - (2. * intersection + smooth) / (union + smooth)  # Dice Loss
    return loss.mean()


def train(args):
    # 設定裝置 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 載入資料集
    train_loader = load_dataset(args.data_path, "train", batch_size=args.batch_size, shuffle=True)
    val_loader = load_dataset(args.data_path, "valid", batch_size=args.batch_size, shuffle=False)

    # # 先檢查 train_loader 的前 5 個 batch，確認 mask 是否正常
    # for i, batch in enumerate(train_loader):
    #     if i >= 5:  # 只顯示 5 張
    #         break
    #     mask = batch["mask"][0].cpu().numpy()  # 取第一張 mask
    #     plt.imshow(mask.squeeze(), cmap="gray")
    #     plt.title(f"Train Batch {i} - Mask")
    #     plt.show()

    # 初始化模型並搬到 GPU
    if args.model == 'resnet34_unet':
        model = ResNet34_UNet( num_classes=1).to(device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)    
        

    # 定義損失函數
    critertion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 紀錄分數最佳的模型
    best_dice = 0.0
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0  # 記錄總 Loss
        train_dice = 0.0  # 記錄 Dice Score

        # 顯示進度條
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in loop:
            images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
            # print(f"DEBUG: image shape = {images.shape}, mask shape = {masks.shape}")

            # 清空梯度
            optimizer.zero_grad()

            # 預測 mask
            outputs = model(images)

            # 計算 Loss
            loss = critertion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            #這樣學習率會逐漸降低，讓 loss 更穩定
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            scheduler.step()

            # print(f"DEBUG: outputs min={outputs.min()}, max={outputs.max()}")
            # print(f"DEBUG: masks min={masks.min()}, max={masks.max()}")
            # print(f"DEBUG: BCE Loss={critertion(outputs, masks).item()}, Dice Loss={dice_loss(outputs, masks).item()}")

            # 記錄 Loss
            train_loss += loss.item()

            # 計算 Dice Score
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            train_dice += dice_score(outputs, masks)

        # 計算 訓練 Dice Score 平均值
        avg_train_dice = train_dice / len(train_loader)
        print(f"Epoch {epoch+1} 訓練 Dice Score: {avg_train_dice}")

        # 執行 evaluate() 在驗證集上測試模型
        avg_val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} 驗證 Dice Score: {avg_val_dice}")

        # 記錄 Dice Score 到 list
        train_dice_scores.append(float(avg_train_dice))  # 確保是 float
        val_dice_scores.append(float(avg_val_dice))  # 確保是 float

        # 儲存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_checkpoint(model, optimizer, epoch+1, args.learning_rate, f"../saved_models/{args.model}_{args.epochs}_best_model.pth")

        

    print("訓練完成！")

    # 繪製 Dice Score 變化
    plot_dice_score(train_dice_scores, val_dice_scores, model_name=args.model)

    # Debug 訊息
    print(f"DEBUG: train_dice_scores = {train_dice_scores}")
    print(f"DEBUG: val_dice_scores = {val_dice_scores}")


def save_checkpoint(model, optimizer, epoch, learning_rate, filename):
    """
    儲存模型權重和優化器狀態
    model: 需要儲存的模型 (UNet)
    optimizer: 優化器 (Adam, SGD, etc.)
    epoch: 訓練了多少 epoch
    learning_rate: 學習率
    filename: 儲存的檔名，預設為 'best_model.pth'
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # 儲存模型參數
        "optimizer_state_dict": optimizer.state_dict(),  # 儲存優化器參數
        "learning_rate": learning_rate,  # 儲存學習率
    }

    torch.save(checkpoint, filename)  # 儲存檔案
    print(f"模型已儲存: {filename} (Epoch {epoch})")

def get_args():
    parser = argparse.ArgumentParser(description='Train on images and target masks')
    parser.add_argument('--download', type=str, default='yes', help='download dataset yes or no')
    parser.add_argument('--model','-m', type=str, default='unet', help='unet or resnet34_unet')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')
    return parser.parse_args()


 

if __name__ == "__main__":

    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #強制 PyTorch 只用 GPU 0
    torch.set_num_threads(1)  # 限制 CPU 使用，強迫使用 GPU

    dataset_path = "../dataset/oxford-iiit-pet"


    if args.download == 'no':
        pass
    else:
        annotations_path = os.path.join(dataset_path, "annotations", "trainval.txt")
        # 檢查數據集是否完整，若缺少則下載
        if not os.path.exists(annotations_path):
            print(f"數據集 `{dataset_path}` 不完整，正在下載...")
            OxfordPetDataset.download(dataset_path)
            print(f"數據集下載完成！")

    # 開始訓練
    train(args)