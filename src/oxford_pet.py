#oxford_pet.py：此檔包含用於處理 Oxford-IIIT Pet 數據集的代碼。它包括數據
#loading、preprocessing 和其他與數據集相關的函數。
import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordPetDataset(torch.utils.data.Dataset):
    #下載圖片dataset
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # 讀取影像 & Mask
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # 使用 Albumentations 變換 image & mask
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return dict(image=image, mask=mask, trimap=trimap)

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        """
        切分訓練測試資料
        """
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):

    #資料前處理
    import torchvision.transforms.functional as TF

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # 讀取影像 & Mask，確保是 PIL Image
        image = Image.open(image_path).convert("RGB")
        trimap = Image.open(mask_path)  # 這是 PIL.Image，需要轉 Tensor
        mask = self._preprocess_mask(np.array(trimap))  # 確保 mask 是二值化

        # 統一大小到 (256, 256)
        image = TF.resize(image, (256, 256))  # 使用 torchvision 重新調整大小
        mask = Image.fromarray(mask).resize((256, 256), Image.NEAREST)  # Mask 使用最近鄰插值
        trimap = trimap.resize((256, 256), Image.NEAREST)  # 這裡修正 trimap 的大小

        # 轉換成 Tensor
        image = TF.to_tensor(image)  # (3, 256, 256)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)  # (256, 256) -> (1, 256, 256)
        trimap = torch.from_numpy(np.array(trimap)).long().unsqueeze(0)  # 轉成 Tensor

        # print(f"DEBUG: image.shape={image.shape}, mask.shape={mask.shape}, trimap.shape={trimap.shape}")  # 檢查形狀

        return dict(image=image, mask=mask, trimap=trimap)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, batch_size=8, shuffle=True):
    """
    轉換影像 & 標註為 PyTorch Tensor
    data_path: 資料集根目錄
    mode: "train" / "valid" / "test"
    batch_size: 批次大小
    shuffle: 是否打亂數據 (訓練時為 True, 測試時為 False)
    """
    assert mode in {"train", "valid", "test"}, "模式應為 'train', 'valid' 或 'test'"
    
    # 定義影像 & Mask 的增強
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()  # 轉換為 PyTorch Tensor
    ])

    # 建立數據集
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode, transform=transform)

    # 建立 DataLoader
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=4)
    
    return dataloader

