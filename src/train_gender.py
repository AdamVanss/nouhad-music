"""Train GenderCNN on UTKFace crops (filename: age_gender_race_date.jpg, gender 0=Boy, 1=Girl)."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.gender_cnn import CLASSES, IMAGE_SIZE, GenderCNN, normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "utkface" / "UTKFace"
CACHE = ROOT / "data" / "utkface" / f"faces_{IMAGE_SIZE}.npz"
CKPT = ROOT / "checkpoints" / "gender_cnn.pt"


def load_faces(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    if CACHE.is_file():
        data = np.load(CACHE)
        return data["images"], data["labels"]
    images, labels = [], []
    for path in sorted(folder.glob("*.jpg")):
        parts = path.name.split("_")
        if len(parts) < 3 or parts[1] not in ("0", "1"):
            continue
        with Image.open(path) as img:
            img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
        labels.append(int(parts[1]))
    images_arr = np.stack(images).transpose(0, 3, 1, 2)
    labels_arr = np.asarray(labels, dtype=np.int64)
    np.savez(CACHE, images=images_arr, labels=labels_arr)
    return images_arr, labels_arr


def augment(x: torch.Tensor) -> torch.Tensor:
    """x float [0,1] (B,3,H,W) on GPU: flip, small shift/zoom/rotation, brightness/contrast."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)
    angle = (torch.rand(b, device=x.device) - 0.5) * (math.pi / 9)
    scale = 1.0 + (torch.rand(b, device=x.device) - 0.5) * 0.2
    shift = (torch.rand(b, 2, device=x.device) - 0.5) * 0.16
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    bright = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    gray = (torch.rand(b, device=x.device) < 0.1).view(b, 1, 1, 1)
    x = torch.where(gray, x.mean(dim=1, keepdim=True).expand_as(x), x)
    return x.clamp(0, 1)


@torch.no_grad()
def accuracy(model, images: torch.Tensor, labels: torch.Tensor, batch: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(images), batch):
        x = normalize(images[i : i + batch])
        logits = model(x) + model(x.flip(-1))
        correct += (logits.argmax(1) == labels[i : i + batch]).sum().item()
    return correct / len(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = load_faces(Path(args.data))
    order = list(range(len(labels)))
    random.Random(0).shuffle(order)
    n_val = len(order) // 10
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_x = torch.from_numpy(images[train_idx]).to(device)
    train_y = torch.from_numpy(labels[train_idx]).to(device)
    val_x = torch.from_numpy(images[val_idx]).to(device)
    val_y = torch.from_numpy(labels[val_idx]).to(device)
    counts = np.bincount(labels, minlength=len(CLASSES))
    print(f"faces {len(labels)}  train {len(train_idx)}  val {n_val}  " + "  ".join(
        f"{name} {c}" for name, c in zip(CLASSES, counts)
    ))

    torch.manual_seed(0)
    model = GenderCNN().to(device)
    print(f"GenderCNN {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15
    )
    scaler = torch.amp.GradScaler("cuda")
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx), device=device)
        running = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = normalize(augment(train_x[idx].float() / 255.0))
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x), train_y[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(idx)
        acc = accuracy(model, val_x, val_y)
        print(f"epoch {epoch:2d}  loss {running / len(train_idx):.4f}  val acc {acc * 100:.2f}%", flush=True)
        if acc > best:
            best = acc
            if acc >= args.min_accuracy:
                CKPT.parent.mkdir(exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "classes": list(CLASSES), "image_size": IMAGE_SIZE,
                     "val_accuracy": acc, "epoch": epoch},
                    CKPT,
                )
    print(f"best val acc {best * 100:.2f}%  " + (f"saved {CKPT}" if best >= args.min_accuracy else "not saved"))


if __name__ == "__main__":
    main()
