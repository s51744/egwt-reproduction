"""
Prepare the cassava-preprocessed dataset (Kaggle: kingofarmy/cassavapreprocessed,
the exact dataset cited by the EGWT paper) into an ImageFolder-style 85/15 split.

Data availability note: the CSV (merged_data.csv) lists 26,220 labeled images
(paper cites 26,337 -- effectively the same dataset, likely a slightly later/earlier
snapshot), but only 24,341 of those actually exist as JPGs in this mirror's
train_images folder (the rest are packaged only as TFRecords in the same download).
We use the 24,341 available JPGs -- a real data-availability constraint of this
specific historical Kaggle mirror, not a deliberate subsampling choice.

The paper additionally applies SMOTE to balance classes before training. SMOTE is
defined over feature vectors, not raw pixels, and the paper does not specify what
feature space it used; reproducing that exact preprocessing step is underspecified.
We substitute a class-weighted loss (weight inversely proportional to class frequency)
during training instead, which addresses the same imbalance problem in a standard,
transparent way. This substitution is documented in the reproduction report.
"""
import shutil
import random
from pathlib import Path
import pandas as pd

CSV = Path(r"C:\Users\Personal\Documents\claude\repro\data\cassava_raw\merged_data.csv")
IMG_DIR = Path(r"C:\Users\Personal\Documents\claude\repro\data\cassava_raw\train_images\train_images")
DST = Path(r"C:\Users\Personal\Documents\claude\repro\data\cassava")
SEED = 42
TRAIN_RATIO = 0.85


def main():
    random.seed(SEED)
    df = pd.read_csv(CSV)
    df["exists"] = df["image_id"].apply(lambda x: (IMG_DIR / x).exists())
    df = df[df["exists"]].reset_index(drop=True)
    print(f"Available labeled images: {len(df)}")

    total_train, total_test = 0, 0
    for cls, group in df.groupby("target"):
        files = group["image_id"].tolist()
        random.shuffle(files)
        n_train = int(round(len(files) * TRAIN_RATIO))
        train_files, test_files = files[:n_train], files[n_train:]

        cls_safe = cls.replace("/", "-")
        (DST / "train" / cls_safe).mkdir(parents=True, exist_ok=True)
        (DST / "test" / cls_safe).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(IMG_DIR / f, DST / "train" / cls_safe / f)
        for f in test_files:
            shutil.copy2(IMG_DIR / f, DST / "test" / cls_safe / f)
        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{cls}: {len(files)} -> {len(train_files)} train / {len(test_files)} test")

    print(f"DONE. train={total_train} test={total_test} total={total_train + total_test}")


if __name__ == "__main__":
    main()
