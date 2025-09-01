import os
import zipfile

# 设置压缩包所在目录
ZIP_DIR = "downloads"   # 这里放你下载的 zip 文件
# 设置仓库根目录（包含 Audio/Tabular/Text/...）
DATASET_ROOT = "."

def extract_zip_files(zip_dir, dataset_root):
    for filename in os.listdir(zip_dir):
        if not filename.endswith(".zip"):
            continue

        # e.g. Audio_easy.zip -> modality="Audio", difficulty="easy"
        name, _ = os.path.splitext(filename)
        try:
            modality, difficulty = name.split("_")
        except ValueError:
            print(f"Skipping {filename}: unexpected naming format")
            continue

        # 目标解压路径
        target_dir = os.path.join(dataset_root, modality, difficulty, "data")
        if not os.path.exists(target_dir):
            print(f"Target directory does not exist: {target_dir}")
            continue

        zip_path = os.path.join(zip_dir, filename)
        print(f"Extracting {zip_path} -> {target_dir}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

if __name__ == "__main__":
    extract_zip_files(ZIP_DIR, DATASET_ROOT)
