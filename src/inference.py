import torch
import argparse
import os
import sys
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

# 解析命令行參數
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks and compute Dice Score for test dataset')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet34_unet'], required=True, help='Select the model architecture')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the test dataset folder (should contain images/ and masks/)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the predicted masks')
    return parser.parse_args()

# 載入模型
def load_model(model_name, model_path, num_classes=1):
    if model_name == 'unet':
        model = UNet()
    elif model_name == 'resnet34_unet':
        model = ResNet34_UNet(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# 預處理
# Resize 為 (256,256)，轉換為 Tensor。
# 正規化為 [-1, 1]。
# unsqueeze(0) 加上 batch 維度，符合模型輸入格式。
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0)  # (1,3,256,256)

# 儲存 Mask
# 移除 batch 與 channel 維度
# 將二值化結果轉換為 0~255 圖片格式。
# 使用 PIL 儲存圖片。
def save_mask(mask, output_path):
    mask = mask.squeeze(0).squeeze(0).numpy()
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).save(output_path)

# Dice 計算
# Flatten 預測與標註，方便計算。
def compute_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)

# 主程式
if __name__ == '__main__':
    args = get_args()
    model = load_model(args.model, args.model_path)

    image_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')
    os.makedirs(args.output_dir, exist_ok=True)

    dice_scores = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + '.png')

        if not os.path.exists(mask_path):
            print(f"⚠️ 缺少 Ground Truth mask: {mask_path}，跳過")
            continue

        # 推理
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.sigmoid(output)
            predicted_mask = (output > 0.5).float()

        # 儲存預測結果
        save_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '_mask.png')
        save_mask(predicted_mask, save_path)

        # 處理 Trimap 標註: 只取 label=1 的區域
        gt_mask = Image.open(mask_path).convert('L').resize((256,256))
        gt_mask = np.array(gt_mask)
        gt_mask = (gt_mask == 1).astype(np.uint8)  # 只取 label==1 的前景作為目標

        pred_mask = predicted_mask.squeeze().numpy().astype(np.uint8)
        dice = compute_dice(pred_mask, gt_mask)
        dice_scores.append(dice)

        print(f"{filename}: Dice Score = {dice:.4f}")

    # 平均 Dice Score
    if dice_scores:
        avg_dice = np.mean(dice_scores)
        print(f"\n🎯 平均 Dice Score: {avg_dice:.4f}")
    else:
        print("⚠️ 沒有計算任何 Dice Score (檢查資料夾結構)")
