# utils.py檔通常包含整個專案中使用的實用程式函數。這些功能
#可能包括可視化工具、指標計算或其他常見任務
import matplotlib.pyplot as plt
import os

def dice_score(pred_mask, gt_mask, smooth=1e-5):
    """
    計算 Dice Score
    Dice Score是一種用來衡量影像分割結果與
    Ground Truth（標註的正確答案）之間相似度的評估指標
    pred_mask: 預測的 Mask (batch, 1, H, W)，值應為 0 或 1，代表每張圖片中 所有 1 的總數
    gt_mask: Ground Truth Mask (batch, 1, H, W)，值應為 0 或 1
    smooth: 避免分母為 0 的平滑項
    return: 平均 Dice Score
    """

    #intersection 計算預測與 Ground Truth 相交的 1 像素總數。
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    intersection = (pred_mask * gt_mask).sum(dim=(2,3))  # 計算交集
    union = pred_mask.sum(dim=(2,3)) + gt_mask.sum(dim=(2,3))

    dice = (2. * intersection + smooth) / (union + smooth)  # 計算 Dice Score
    return dice.mean().item()  # 回傳 batch 平均



def plot_dice_score(train_dice, val_dice, model_name, save_dir="./result"):
    epochs = range(1, len(train_dice) + 1)

    # 自動命名圖片檔案，包含 epoch 數值範圍
    filename = f"{model_name}_dice_score_{epochs[0]}-{epochs[-1]}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_dice, label="Train Dice Score", marker="o")
    plt.plot(epochs, val_dice, label="Validation Dice Score", marker="s")

    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title(f"{model_name} Dice Score Trend")
    plt.legend()
    plt.grid(True)

    # 存圖
    os.makedirs(save_dir, exist_ok=True)  # 確保目錄存在
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 儲存圖片
    print(f"Dice Score 圖已儲存至: {save_path}")

    plt.close()  # 關閉圖表，避免記憶體累積


   

