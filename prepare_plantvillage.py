"""
Prepare PlantVillage dataset per the EGWT paper spec:
- Use the 'color' subset (54,305 images, 38 classes)
- 85% train / 15% test split, stratified per class, deterministic seed
- Copies files into repro/data/plantvillage/{train,test}/<class>/ for fast local NTFS access
"""
import os
import random
import shutil
from pathlib import Path

SRC = Path(r"\\wsl.localhost\Ubuntu\home\fishlord\dataset\plantvillage dataset\color")
DST = Path(r"C:\Users\Personal\Documents\claude\repro\data\plantvillage")
SEED = 42
TRAIN_RATIO = 0.85

def main():
    random.seed(SEED)
    classes = sorted([d.name for d in SRC.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes")

    total_train, total_test = 0, 0
    for cls in classes:
        cls_dir = SRC / cls
        files = sorted([f.name for f in cls_dir.iterdir() if f.is_file()])
        random.shuffle(files)
        n_train = int(round(len(files) * TRAIN_RATIO))
        train_files = files[:n_train]
        test_files = files[n_train:]

        (DST / "train" / cls).mkdir(parents=True, exist_ok=True)
        (DST / "test" / cls).mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy2(cls_dir / f, DST / "train" / cls / f)
        for f in test_files:
            shutil.copy2(cls_dir / f, DST / "test" / cls / f)

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{cls}: {len(files)} total -> {len(train_files)} train / {len(test_files)} test")

    print(f"\nDONE. Total train={total_train}, test={total_test}, grand_total={total_train+total_test}")

if __name__ == "__main__":
    main()
