"""
Prepare the cassava leaf disease dataset into an ImageFolder-style 85/15 split.

Source: Kaggle nirmalsankalana/cassava-leaf-disease-classification (a JPEG mirror
of the 2020 "Cassava Leaf Disease Classification" competition data), 5 classes,
21,397 images total -- matching the user-confirmed paper figure of 21,397 exactly.
This replaces the repo's original source (kingofarmy/cassavapreprocessed, whose
own script comment says it was chasing a different paper-cited count of 26,337
and only had 24,341 of those actually available as JPGs) -- the 21,397 dataset
is a closer, cleaner match to what the paper is quoted as stating.

The paper additionally applies SMOTE to balance classes before training. SMOTE is
defined over feature vectors, not raw pixels, and the paper does not specify what
feature space it used; reproducing that exact preprocessing step is underspecified.
We substitute a class-weighted loss (weight inversely proportional to class frequency)
during training instead, which addresses the same imbalance problem in a standard,
transparent way. This substitution is documented in the reproduction report.
"""
import random
import shutil
from pathlib import Path

SRC = Path(r"E:\plant_disease\cassava_download\data")
DST = Path(r"E:\plant_disease\egwt-reproduction\data\cassava")
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

    print(f"\nDONE. Total train={total_train}, test={total_test}, grand_total={total_train + total_test}")


if __name__ == "__main__":
    main()
