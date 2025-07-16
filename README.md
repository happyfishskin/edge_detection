# Oxford-IIIT Pet 影像分割專案

這是一個使用 PyTorch 實現的影像分割專案，旨在對 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) 進行寵物主體分割。專案實現了兩種深度學習模型：**UNet** 和 **ResNet34-UNet**。

## 專案特色

* **兩種模型架構**：
    * **UNet**：經典的影像分割架構，適用於生物醫學影像。
    * **ResNet34-UNet**：使用預訓練的 ResNet-34 作為編碼器，能更好地提取特徵，通常效果更佳。
* **資料增強**：使用 `albumentations` 函式庫進行多種資料增強，提升模型的泛化能力。
* **完整流程**：包含資料下載、模型訓練、驗證、儲存、以及最終的推理與評估。
* **視覺化結果**：訓練完成後會自動繪製並儲存訓練與驗證的 Dice Score 趨勢圖。

## 檔案結構說明

```
.
├── saved_models/         # 儲存訓練好的模型權重 (.pth)
├── result/               # 儲存訓練結果圖表 (dice_score.png)
├── test/                 # 存放整理好的測試資料集
│   ├── images/
│   └── masks/
├── dataset/
│   └── oxford-iiit-pet/  # 原始資料集
├── src/
│   ├── models/
│   │   ├── unet.py             # UNet 模型架構
│   │   └── resnet34_unet.py    # ResNet34-UNet 模型架構
│   ├── download.py         # (可選) 獨立下載資料集
│   ├── oxford_pet.py       # 資料集處理與載入
│   ├── train.py            # 主要訓練腳本
│   ├── evaluate.py         # 驗證集評估邏輯
│   ├── inference.py        # 使用訓練好的模型進行推理
│   ├── testdata.py         # 將官方測試集整理至 test/ 資料夾
│   └── utils.py            # 工具函式 (Dice Score 計算、繪圖)
└── README.md             # 本說明文件
```

## 環境設定

1.  **安裝必要的 Python 函式庫**：
    建議您建立一個虛擬環境。

    ```bash
    pip install torch torchvision
    pip install Pillow numpy matplotlib tqdm albumentations
    ```

## 使用教學

### 1. 下載並準備資料集

您可以透過以下兩種方式之一來準備資料：

* **方法一：執行 `train.py` 自動下載**
    `train.py` 腳本會自動檢查資料集是否存在，如果不存在將會自動下載。

* **方法二：手動執行 `download.py`**
    ```bash
    python src/download.py
    ```
    這會將 Oxford-IIIT Pet 資料集下載到 `dataset/oxford-iiit-pet` 路徑下。

### 2. 準備測試資料夾

執行 `testdata.py` 將官方測試集中的圖片和標籤複製到獨立的 `test/` 資料夾中，方便後續進行推理。

```bash
python src/testdata.py
```

### 3. 訓練模型

執行 `train.py` 來開始訓練。您可以透過命令行參數來選擇模型、設定超參數。

**UNet 模型訓練範例：**
```bash
python src/train.py --model unet --epochs 20 --batch_size 8 --learning_rate 1e-4 --data_path ../dataset/oxford-iiit-pet
```

**ResNet34-UNet 模型訓練範例：**
```bash
python src/train.py --model resnet34_unet --epochs 20 --batch_size 8 --learning_rate 1e-4 --data_path ../dataset/oxford-iiit-pet
```

* `--model`: 選擇 `unet` 或 `resnet34_unet`。
* `--epochs`: 訓練的週期數。
* `--batch_size`: 批次大小。
* `--learning_rate`: 學習率。
* `--data_path`: 資料集根目錄的路徑。

訓練過程中，腳本會自動評估驗證集上的 Dice Score，並將分數最高的模型權重儲存於 `saved_models/` 資料夾下。訓練結束後，結果圖表會儲存在 `result/` 資料夾。

### 4. 進行推理與評估

訓練完成後，使用 `inference.py` 在 `test/` 資料夾中的測試集上進行推理，並計算平均 Dice Score。

**推理範例：**
```bash
python src/inference.py --model unet --model_path ../saved_models/unet_20_best_model.pth --data_dir ../test --output_dir ../predictions
```

* `--model`: 選擇您訓練時使用的模型。
* `--model_path`: 指向您儲存的最佳模型權重檔案 (`.pth`)。
* `--data_dir`: 包含 `images` 和 `masks` 的測試資料夾路徑 (即 `test/`)。
* `--output_dir`: 用於儲存模型預測出的 Mask 圖片。

腳本會逐一處理測試圖片，印出每張圖片的 Dice Score，並在最後計算所有測試圖片的平均 Dice Score。
