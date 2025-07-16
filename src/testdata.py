import os
import shutil

# 資料集路徑（請根據實際路徑修改）
dataset_root = "../dataset/oxford-iiit-pet"
test_txt_path = os.path.join(dataset_root, "annotations", "test.txt")
images_dir = os.path.join(dataset_root, "images")
masks_dir = os.path.join(dataset_root, "annotations", "trimaps")
output_dir = "../test"

# 建立目標資料夾
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# 讀取 test.txt
with open(test_txt_path, "r") as f:
    lines = f.readlines()
    test_files = [line.strip().split()[0] for line in lines if line.strip()]


# 複製檔案
for filename in test_files:
    image_file = os.path.join(images_dir, filename + ".jpg")
    mask_file = os.path.join(masks_dir, filename + ".png")

    target_image = os.path.join(output_dir, "images", filename + ".jpg")
    target_mask = os.path.join(output_dir, "masks", filename + ".png")

    if os.path.exists(image_file) and os.path.exists(mask_file):
        shutil.copy(image_file, target_image)
        shutil.copy(mask_file, target_mask)
        print(f"✅ 已複製: {filename}")
    else:
        print(f"⚠️ 缺少檔案: {filename}")

print("\n🎯 Test 資料夾整理完成！")
print(f"Test 資料夾位置: {output_dir}")
