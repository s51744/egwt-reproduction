"""
Reorganize the Kaggle ILSVRC2012 competition download into a standard
torchvision.datasets.ImageFolder layout:
  imagenet/train/<synset>/*.JPEG   (already in this layout after extraction)
  imagenet/val/<synset>/*.JPEG     (built here from the flat val folder +
                                     LOC_val_solution.csv, which maps each val image
                                     filename to its ground-truth synset)
"""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted_root", required=True,
                     help="Root of the extracted competition zip (contains ILSVRC/, LOC_val_solution.csv, etc.)")
    ap.add_argument("--out_dir", default=r"C:\Users\Personal\Documents\claude\repro\data\imagenet")
    ap.add_argument("--move", action="store_true",
                     help="Move files instead of copy (faster, saves disk, but consumes the source)")
    args = ap.parse_args()

    root = Path(args.extracted_root)
    out = Path(args.out_dir)

    train_src = root / "ILSVRC" / "Data" / "CLS-LOC" / "train"
    val_src = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    val_solution = root / "LOC_val_solution.csv"

    print(f"train_src exists: {train_src.exists()}  val_src exists: {val_src.exists()}  "
          f"val_solution exists: {val_solution.exists()}")

    (out / "train").parent.mkdir(parents=True, exist_ok=True)
    if not (out / "train").exists():
        # Windows symlinks require elevated privileges (SeCreateSymbolicLinkPrivilege),
        # which this shell doesn't have -- move (rename) instead, which is free (same
        # volume) and needs no special permission.
        print("Moving train/ into place (rename, same volume -- no extra disk cost)...")
        shutil.move(str(train_src), str(out / "train"))
        print("Moved train/ into place")

    # Build val/<synset>/ from the flat val folder + solution CSV
    (out / "val").mkdir(parents=True, exist_ok=True)
    with open(val_solution, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            image_id = row["ImageId"]
            synset = row["PredictionString"].split()[0]  # first token is the synset id
            (out / "val" / synset).mkdir(exist_ok=True)
            src_file = val_src / f"{image_id}.JPEG"
            dest_file = out / "val" / synset / f"{image_id}.JPEG"
            if src_file.exists() and not dest_file.exists():
                shutil.move(str(src_file), str(dest_file))  # symlinks need elevated Windows privileges
                n += 1
    print(f"DONE. val images organized: {n}")


if __name__ == "__main__":
    main()
