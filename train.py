import math
import random
import argparse
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    ablation="full",
    module_ablation="full",
):
    model.train()
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"].to(device)

        # 训练阶段仅在存在音频分支时加微量噪声
        if ablation != "text":
            noise_std = 0.02
            audio = audio + noise_std * torch.randn_like(audio)

        optimizer.zero_grad()

        out = model(
            audio,
            local,
            global_t,
            mask,
            ablation=ablation,
            module_ablation=module_ablation,
        )

        logits = out["logits"]
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    ablation="full",
    module_ablation="full",
):
    model.eval()
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"].to(device)

        out = model(
            audio,
            local,
            global_t,
            mask,
            ablation=ablation,
            module_ablation=module_ablation,
        )

        logits = out["logits"]
        loss = ce(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)

    return total_loss / total_n, total_correct / total_n


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        e = epoch
        if e < warmup_epochs:
            return float(e + 1) / float(warmup_epochs)
        progress = float(e - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_txt", type=str, default="dataset/list_NCMMSC2021/train.txt")
    parser.add_argument("--val_txt", type=str, default="dataset/list_NCMMSC2021/test.txt")
    parser.add_argument("--ckpt", type=str, default="best_gate_cls.pt")

    # 调参相关
    parser.add_argument("--base_lr", type=float, default=4e-3)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=96)
    parser.add_argument("--num_factors", type=int, default=12)

    # 模态消融：full / audio / text
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "audio", "text"]
    )

    # 模块消融：base / lgsga / atcr / full
    parser.add_argument(
        "--module_ablation",
        type=str,
        default="full",
        choices=["base", "lgsga", "atcr", "full"]
    )

    parser.add_argument("--max_epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = GateDataset(args.train_txt)
    val_set = GateDataset(args.val_txt)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # 三分类写 3；英文二分类时改成 2
    num_classes = 3

    model = GateAggWithBiGRUClassifier(
        audio_dim=768,
        text_dim=768,
        rnn_hidden=args.rnn_hidden,
        hidden_gate=args.hidden_gate,
        hidden_main=args.hidden_main,
        num_main_classes=num_classes,
        num_factors=args.num_factors,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=5e-4
    )

    warmup_epochs = 5
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs,
        args.max_epochs
    )

    best_acc = 0.0
    best_epoch = 0

    print("=" * 90)
    print(f"seed={args.seed}")
    print(f"train_txt={args.train_txt}")
    print(f"val_txt={args.val_txt}")
    print(f"ablation={args.ablation}")
    print(f"module_ablation={args.module_ablation}")
    print(f"ckpt={args.ckpt}")
    print("=" * 90)

    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ablation=args.ablation,
            module_ablation=args.module_ablation,
        )

        val_loss, val_acc = eval_one_epoch(
            model,
            val_loader,
            device,
            ablation=args.ablation,
            module_ablation=args.module_ablation,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"seed={args.seed} "
            f"ablation={args.ablation} "
            f"module_ablation={args.module_ablation} "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

            save_path = args.ckpt
            ckpt_dir = os.path.dirname(save_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)

            torch.save(model.state_dict(), save_path)
            print(
                f"  -> save best model to {save_path}, "
                f"best_val_acc={best_acc:.4f} at epoch {best_epoch}"
            )
        else:
            if (epoch + 1) - best_epoch >= args.patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, "
                    f"best_val_acc={best_acc:.4f} at epoch {best_epoch}"
                )
                break

    print("=" * 90)
    print(
        f"Training finished. "
        f"Best val_acc={best_acc:.4f} at epoch {best_epoch} | "
        f"ablation={args.ablation} | module_ablation={args.module_ablation}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()