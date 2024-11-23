import os
import random
import shutil
import csv
from glob import glob
from tqdm import tqdm

# 파일 리스트 가져오기
image_files = glob('/home/jmkim/dev/efficientnet/images_000/**/*.jpg', recursive=True)

# Step 1: `filtered_result2.csv`에서 id와 hierarchical_label로 딕셔너리 생성
label_dict = {}
with open('final_result.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
    header = rows[0]
    id_index = header.index("id")
    label_index = header.index("hierarchical_label")

    for row in rows[1:]:
        if len(row) > label_index and row[label_index]:
            label_dict[row[id_index]] = row[label_index]

# Step 2: 적합한 디렉토리 구조로 이미지 파일 복사
output_dir = "/home/jmkim/dev/efficientnet/output_dataset"  # 새 디렉토리 루트
os.makedirs(output_dir, exist_ok=True)

for image_path in image_files:
    # 이미지 이름에서 확장자를 제거하고 id 추출
    image_id = os.path.basename(image_path).replace('.jpg', '')

    if image_id in label_dict:
        # hierarchical_label 기반 폴더 경로 생성
        label_name = label_dict[image_id]
        target_dir = os.path.join(output_dir, label_name)
        os.makedirs(target_dir, exist_ok=True)

        # 이미지 복사 또는 이동
        target_path = os.path.join(target_dir, os.path.basename(image_path))
        shutil.copy(image_path, target_path)  # 파일 복사 (shutil.move로 변경하면 이동)
        # print(f"Copied: {image_path} -> {target_path}")

print("Dataset restructuring complete!")

# Step 1: 기존 데이터 디렉토리
input_dir = "/home/jmkim/dev/efficientnet/output_dataset"  # 생성된 디렉토리
output_dir = "/home/jmkim/dev/efficientnet/final_dataset"  # train/val 폴더를 포함할 최종 디렉토리
os.makedirs(output_dir, exist_ok=True)

# Step 2: train/val 디렉토리 생성
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Step 3: 각 클래스 폴더 처리
classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

for cls in tqdm(classes, desc="Processing classes"):
    class_dir = os.path.join(input_dir, cls)
    images = glob(os.path.join(class_dir, "*.jpg"))

    # 셔플 후 분할
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)  # 80% split index
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 클래스별 train/val 폴더 생성
    train_class_dir = os.path.join(train_dir, cls)
    val_class_dir = os.path.join(val_dir, cls)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # 이미지 이동 (train 폴더)
    for img in train_images:
        shutil.move(img, os.path.join(train_class_dir, os.path.basename(img)))

    # 이미지 이동 (val 폴더)
    for img in val_images:
        shutil.move(img, os.path.join(val_class_dir, os.path.basename(img)))

print(f"Dataset successfully split into train and val folders at {output_dir}")

from torchvision import datasets

# 데이터셋 디렉토리
dataset_path = "/home/jmkim/dev/efficientnet/final_dataset/train"

# ImageFolder로 데이터셋 로드
dataset = datasets.ImageFolder(dataset_path)

# 클래스 이름과 인덱스 확인
class_to_idx = dataset.class_to_idx  # {클래스 이름: 인덱스}
idx_to_class = {v: k for k, v in class_to_idx.items()}  # {인덱스: 클래스 이름}

print("클래스 인덱스 매핑:")
for idx, class_name in idx_to_class.items():
    print(f"{idx}: {class_name}")

# 클래스 매핑 저장
mapping_file = "class_mapping.txt"
with open(mapping_file, "w") as f:
    for idx, class_name in idx_to_class.items():
        f.write(f"{idx}: {class_name}\n")

print(f"클래스 매핑 파일이 생성되었습니다: {mapping_file}")