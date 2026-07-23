"""
Prepare the tomato leaves dataset (Kaggle: ashishmotwani/tomato, the exact dataset
cited by the EGWT paper) into an 85/15 train/test split per the paper's spec.
The Kaggle download ships as train/valid (25,851 + 6,683 = 32,534 images, matching
the paper's reported 32,531 almost exactly); we merge both and re-split 85/15,
stratified per class, to match the paper's stated protocol exactly.
"""
import shutil
import random
from pathlib import Path

SRC_TRAIN = Path(r"C:\Users\Personal\Documents\claude\repro\data\tomato_raw\train")
SRC_VALID = Path(r"C:\Users\Personal\Documents\claude\repro\data\tomato_raw\valid")
DST = Path(r"C:\Users\Personal\Documents\claude\repro\data\tomato")
SEED = 42
TRAIN_RATIO = 0.85


def main():
    random.seed(SEED)
    classes = sorted([d.name for d in SRC_TRAIN.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes")

    total_train, total_test = 0, 0
    for cls in classes:
        files = []
        for src in (SRC_TRAIN, SRC_VALID):
            cls_dir = src / cls
            if cls_dir.exists():
                files += [(cls_dir / f.name) for f in cls_dir.iterdir() if f.is_file()]
        random.shuffle(files)
        n_train = int(round(len(files) * TRAIN_RATIO))
        train_files, test_files = files[:n_train], files[n_train:]

        (DST / "train" / cls).mkdir(parents=True, exist_ok=True)
        (DST / "test" / cls).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, DST / "train" / cls / f.name)
        for f in test_files:
            shutil.copy2(f, DST / "test" / cls / f.name)
        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{cls}: {len(files)} -> {len(train_files)} train / {len(test_files)} test")

    print(f"DONE. train={total_train} test={total_test} total={total_train + total_test}")


if __name__ == "__main__":
    main()
