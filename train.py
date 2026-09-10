"""
Training script for EGWT, mirroring the paper's fine-tuning protocol (Section IV):
  - Adam optimizer, decreasing LR (decline ratio 0.9 -- interpreted as an exponential
    decay factor applied per epoch, since the paper does not specify the schedule's
    exact functional form beyond "decline ratio of 0.9 and a stabilization ratio of
    0.1"; we use: lr_t = max(lr0 * 0.9^t, lr0 * 0.1))
  - 224x224 input
  - stage1+stage2 frozen, fine-tune from stage3 onward (Section IV.C)
  - random cropping, rotation, flipping augmentation
  - stage-3 EfficientNet-projection blocks warm-started from real ImageNet weights;
    everything else randomly initialized (no full EGWT ImageNet pretraining

Compute-budget deviation (documented, user-approved): epoch count is reduced from
whatever the paper implies for convergence, to fit within the overall 6-hour budget
across three datasets. Exact epoch counts used are logged per run.
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


def build_loaders(data_dir, batch_size, num_workers=6):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(Path(data_dir) / "test", transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, test_loader, len(train_ds.classes)


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mat = pred.eq(y.view(1, -1).expand_as(pred))
        for k in topk:
            correct[k] += correct_mat[:k].reshape(-1).float().sum().item()
        total += y.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    acc = {k: correct[k] / total for k in topk}
    return acc, torch.cat(all_preds), torch.cat(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--pretrained_effnet", action="store_true", default=True)
    # Two competing readings of the paper's fine-tuning protocol, both reproduced for
    # comparison: "stage12" freezes stage1+2 only (stage3+head trainable); "head_only"
    # freezes everything but the final linear layer (paper's transfer-learning section,
    # verbatim); "none" is full fine-tuning (no freeze) -- kept as the pre-existing
    # baseline this repo already had results for.
    ap.add_argument("--freeze_mode", choices=["none", "stage12", "head_only"], default="stage12")
    ap.add_argument("--imagenet_ckpt", default=None,
                     help="Path to a full-model ImageNet-pretrained state_dict (egwt_imagenet_best.pt from "
                          "pretrain_imagenet.py). If given, this replaces the pretrained_effnet compute-budget "
                          "compromise with genuine full-model ImageNet pretraining, matching the paper's own "
                          "protocol (Section III.B.5).")
    ap.add_argument("--class_weighted_loss", action="store_true",
                     help="Inverse-frequency class weighting on CrossEntropyLoss -- documented substitute for "
                          "the paper's SMOTE oversampling (SMOTE's feature space isn't specified), used for "
                          "cassava's severe class imbalance. Was documented in prepare_cassava.py but never "
                          "actually wired into the loss until now.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_classes = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"[{args.dataset_name}] classes={n_classes} train_batches={len(train_loader)} test_batches={len(test_loader)}")

    if args.imagenet_ckpt:
        model = EGWT(num_classes=1000, pretrained_effnet=False).to(device)
        state = torch.load(args.imagenet_ckpt, map_location=device)
        model.load_state_dict(state)
        model.head = torch.nn.Linear(1024, n_classes).to(device)  # replace 1000-way head with this dataset's
        print(f"Loaded full ImageNet-pretrained weights from {args.imagenet_ckpt}")
    else:
        model = EGWT(num_classes=n_classes, pretrained_effnet=args.pretrained_effnet).to(device)
    if args.freeze_mode == "stage12":
        model.freeze_stage12()
    elif args.freeze_mode == "head_only":
        model.freeze_all_but_head()
    print(f"Total params: {count_params(model)/1e6:.2f}M, trainable: {count_params(model, True)/1e6:.2f}M")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: max(0.9 ** epoch, 0.1))
    if args.class_weighted_loss:
        train_ds = train_loader.dataset
        counts = torch.bincount(torch.tensor(train_ds.targets), minlength=n_classes).float()
        weight = (counts.sum() / (n_classes * counts)).to(device)
        print("class weights:", weight.tolist())
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    # bf16 autocast instead of fp16+GradScaler: this model has 646 individual
    # parameter tensors (G-MHA/G-MLP's per-group nn.ModuleList linears fragment
    # what would normally be a few large weight matrices into hundreds of small
    # ones). GradScaler's inf/nan check walks every parameter's gradient tensor
    # individually each step (a host-device sync per tensor) -- with this many
    # small tensors that serializes and dominates step time (measured: 30+ min
    # stuck on a single optimizer.step() call, GPU at 99% util but not actually
    # progressing -- the same "high util, low real throughput" fragmented-kernel
    # signature diagnosed for ConViTX's 16-head attention earlier in this
    # project). bf16 has fp32's exponent range so it never needs loss scaling,
    # which removes GradScaler and this bottleneck entirely. Safe on this GPU
    # (RTX 4070 Ti / Ada Lovelace, confirmed bf16-capable).

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.dataset_name}_{args.freeze_mode}"
    log_path = out_dir / f"egwt_{run_tag}_log.jsonl"
    best_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        acc, preds, labels = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch, "train_loss": running_loss / max(n_batches, 1),
            "top1": acc[1], "top5": acc.get(5, None), "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": elapsed,
        }
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{run_tag}] epoch {epoch}: loss={rec['train_loss']:.4f} "
              f"top1={acc[1]*100:.2f}% top5={acc.get(5,0)*100:.2f}% time={elapsed:.1f}s")

        if acc[1] > best_acc:
            best_acc = acc[1]
            torch.save(model.state_dict(), out_dir / f"egwt_{run_tag}_best.pt")

    # final metrics: precision/recall/f1 (macro) from the last epoch's predictions
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    summary = {
        "dataset": args.dataset_name, "freeze_mode": args.freeze_mode, "n_classes": n_classes,
        "best_top1": best_acc, "final_top1": acc[1], "final_top5": acc.get(5, None),
        "final_precision_macro": p, "final_recall_macro": r, "final_f1_macro": f1,
        "total_params_M": count_params(model) / 1e6,
        "trainable_params_M": count_params(model, True) / 1e6,
        "epochs_run": args.epochs,
    }
    with open(out_dir / f"egwt_{run_tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
