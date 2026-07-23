"""
Pretrain EGWT from scratch on ImageNet-1k, per the paper's own protocol (Section
III.B.5): "the EGWT model is pre-trained from scratch using the ImageNet dataset.
Subsequently, the pre-trained EGWT model is fine-tuned using the crop disease
datasets." This replaces the earlier compute-budget compromise (real EfficientNet-B0
weights seeded into stage 3 only, everything else randomly initialized and trained
directly on the small crop datasets) with genuine full-model ImageNet pretraining.

The paper does not give ImageNet-pretraining-specific hyperparameters separately from
its overall "training process" description (Adam, decreasing LR with decline ratio 0.9
/ stabilization ratio 0.1) -- we reuse that same schedule here rather than substituting
an unstated convention, even though it floors to 10% of the initial LR relatively early
(~22 epochs) or a long run. Batch size and augmentation specifics for the ImageNet
stage are likewise not given; we use RandomResizedCrop+flip (the standard, near-
universal ImageNet recipe) and a batch size chosen for GPU throughput, documented in
the run log.

Supports resume-from-checkpoint since a many-epoch ImageNet run is expected to span
multiple sessions.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EGWT, count_params

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_loaders(data_dir, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader, len(train_ds.classes)


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mat = pred.eq(y.view(1, -1).expand_as(pred))
        for k in topk:
            correct[k] += correct_mat[:k].reshape(-1).float().sum().item()
        total += y.size(0)
    return {k: correct[k] / total for k in topk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"C:\Users\Personal\Documents\claude\repro\data\imagenet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir", default=r"C:\Users\Personal\Documents\claude\repro\checkpoints")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, n_classes = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"ImageNet: classes={n_classes} train_batches={len(train_loader)} val_batches={len(val_loader)}")

    model = EGWT(num_classes=n_classes, pretrained_effnet=False).to(device)  # genuine from-scratch
    print(f"Total params: {count_params(model)/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: max(0.9 ** epoch, 0.1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "egwt_imagenet_ckpt.pt"
    log_path = out_dir / "egwt_imagenet_log.jsonl"

    start_epoch = 0
    best_top1 = 0.0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed from epoch {start_epoch} (best_top1={best_top1:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
            if i % 500 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_loader)} loss={loss.item():.4f}")
        scheduler.step()

        acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        rec = {"epoch": epoch, "train_loss": running_loss / max(n_batches, 1),
               "val_top1": acc[1], "val_top5": acc.get(5), "lr": scheduler.get_last_lr()[0],
               "epoch_time_sec": elapsed}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[imagenet] epoch {epoch}: loss={rec['train_loss']:.4f} "
              f"top1={acc[1]*100:.2f}% top5={acc.get(5,0)*100:.2f}% time={elapsed/60:.1f}min")

        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save(model.state_dict(), out_dir / "egwt_imagenet_best.pt")

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_top1": best_top1,
        }, ckpt_path)

    print(f"DONE. best_top1={best_top1:.4f}")


if __name__ == "__main__":
    main()
