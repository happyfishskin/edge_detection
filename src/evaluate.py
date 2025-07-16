#evaluate.py：此檔可能處理模型評估。它包括用於評估模型的函數
#驗證性能。

import torch
from utils import dice_score  # 確保 dice_score 正確導入

def evaluate(model, val_loader, device):
    model.eval()  # 設定為評估模式
    dice_score_total = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            # 計算 Dice Score
            batch_dice_score = dice_score(outputs, masks)
            dice_score_total += batch_dice_score

            # 每 10 個 batch 印出一次 DEBUG 訊息
            if batch_idx % 10 == 0:
                print(f"DEBUG: Eval Batch {batch_idx}/{len(val_loader)} - Dice Score: {batch_dice_score:.6f}")

    # 確保不會被 0 除錯誤
    avg_dice_score = dice_score_total / max(len(val_loader), 1)
    print(f"DEBUG: Final Evaluation Dice Score: {avg_dice_score:.6f}")

    return avg_dice_score
