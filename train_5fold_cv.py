import os
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator != 0 else 0.0


def compute_metrics_from_labels(y_true, y_pred, num_classes):
    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    total = cm.sum()
    acc = safe_div(np.trace(cm), total)

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(int(support))

    metrics = {
        "acc": acc,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class_id": i,
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1": float(f1s[i]),
                "support": supports[i],
            }
            for i in range(num_classes)
        ],
    }
    return metrics



def train_one_epoch(model, loader, optimizer, device, noise_std=0.02):
    model.train()
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"].to(device)

        if noise_std > 0:
            audio = audio + noise_std * torch.randn_like(audio)

        optimizer.zero_grad()
        out = model(audio, local, global_t, mask)
        logits = out["logits"]
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
    }


@torch.no_grad()
def eval_one_epoch(model, loader, device, num_classes):
    model.eval()
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss, total_correct, total_n = 0.0, 0, 0
    all_true, all_pred = [], []

    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"].to(device)

        out = model(audio, local, global_t, mask)
        logits = out["logits"]
        loss = ce(logits, y)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics_from_labels(all_true, all_pred, num_classes)
    metrics["loss"] = total_loss / max(1, total_n)
    metrics["acc"] = total_correct / max(1, total_n)
    metrics["y_true"] = all_true
    metrics["y_pred"] = all_pred
    return metrics



def make_result_tag(args):
    lr_str = format(args.base_lr, "g")
    return f"s{args.seed}_lr{lr_str}_rh{args.rnn_hidden}_hg{args.hidden_gate}_hm{args.hidden_main}_k{args.num_factors}"



def train_one_fold(args, fold_id, device):
    set_seed(args.seed)

    train_txt = Path(args.fold_dir) / args.train_pattern.format(fold=fold_id)
    val_txt = Path(args.fold_dir) / args.val_pattern.format(fold=fold_id)

    if not train_txt.exists():
        raise FileNotFoundError(f"训练列表不存在: {train_txt}")
    if not val_txt.exists():
        raise FileNotFoundError(f"验证列表不存在: {val_txt}")

    train_set = GateDataset(str(train_txt))
    val_set = GateDataset(str(val_txt))

    g = torch.Generator()
    g.manual_seed(args.seed + fold_id)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = GateAggWithBiGRUClassifier(
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        rnn_hidden=args.rnn_hidden,
        hidden_gate=args.hidden_gate,
        hidden_main=args.hidden_main,
        num_main_classes=args.num_classes,
        num_factors=args.num_factors,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.max_epochs)

    save_dir = Path(args.ckpt_root) / f"fold_{fold_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_gate_cls.pt"

    best_acc = -1.0
    best_epoch = -1
    stale_epochs = 0

    print(f"\n{'=' * 90}")
    print(f"开始训练 Fold {fold_id}")
    print(f"train_txt = {train_txt}")
    print(f"val_txt   = {val_txt}")
    print(f"{'=' * 90}")

    for epoch in range(1, args.max_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, noise_std=args.noise_std)
        val_stats = eval_one_epoch(model, val_loader, device, args.num_classes)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Fold {fold_id} | Epoch {epoch:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.4f}, "
            f"val_prec={val_stats['precision_macro']:.4f}, val_rec={val_stats['recall_macro']:.4f}, "
            f"val_f1={val_stats['f1_macro']:.4f}"
        )

        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), str(best_ckpt))
            print(f"  -> 保存当前 fold 最优模型: {best_ckpt}")
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"  -> Early stopping at epoch {epoch}, best epoch = {best_epoch}, best acc = {best_acc:.4f}")
                break

    # 用最佳权重重新评估，确保最终指标与保存的模型一致
    model.load_state_dict(torch.load(str(best_ckpt), map_location=device))
    best_metrics = eval_one_epoch(model, val_loader, device, args.num_classes)

    fold_result = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_ckpt": str(best_ckpt),
        "num_train": len(train_set),
        "num_val": len(val_set),
        "acc": best_metrics["acc"],
        "precision_macro": best_metrics["precision_macro"],
        "recall_macro": best_metrics["recall_macro"],
        "f1_macro": best_metrics["f1_macro"],
        "loss": best_metrics["loss"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "per_class": best_metrics["per_class"],
        "y_true": best_metrics["y_true"],
        "y_pred": best_metrics["y_pred"],
    }

    print(
        f"[Fold {fold_id} 最终结果] "
        f"acc={fold_result['acc']:.4f}, "
        f"prec={fold_result['precision_macro']:.4f}, "
        f"rec={fold_result['recall_macro']:.4f}, "
        f"f1={fold_result['f1_macro']:.4f}, "
        f"best_epoch={fold_result['best_epoch']}"
    )

    return fold_result



def summarize_cv_results(fold_results, num_classes):
    keys = ["acc", "precision_macro", "recall_macro", "f1_macro", "loss"]
    summary = {}
    for key in keys:
        values = [fr[key] for fr in fold_results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "values": [float(v) for v in values],
        }

    all_true, all_pred = [], []
    for fr in fold_results:
        all_true.extend(fr["y_true"])
        all_pred.extend(fr["y_pred"])

    overall = compute_metrics_from_labels(all_true, all_pred, num_classes)
    overall["num_samples"] = len(all_true)
    return summary, overall



def main():
    parser = argparse.ArgumentParser(description="自动运行五折交叉验证，并汇总 Accuracy / Precision / Recall / F1")

    # 数据相关
    parser.add_argument("--fold_dir", type=str, default="dataset/list_ADReSSo_5fold")
    parser.add_argument("--train_pattern", type=str, default="train_fold_{fold}.txt")
    parser.add_argument("--val_pattern", type=str, default="test_fold_{fold}.txt")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=2)

    # 模型与训练超参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--base_lr", type=float, default=0.003)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=96)
    parser.add_argument("--num_factors", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=45)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--noise_std", type=float, default=0.02)

    # 输出相关
    parser.add_argument("--ckpt_root", type=str, default="")
    parser.add_argument("--result_json", type=str, default="")

    args = parser.parse_args()

    tag = make_result_tag(args)
    if not args.ckpt_root:
        args.ckpt_root = str(Path("ckpts") / f"cv5_{tag}")
    if not args.result_json:
        args.result_json = str(Path(args.ckpt_root) / f"cv5_results_{tag}.json")

    Path(args.ckpt_root).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    print(f"结果保存目录: {args.ckpt_root}")
    print(f"结果JSON: {args.result_json}")

    fold_results = []
    for fold_id in range(1, args.n_folds + 1):
        fold_result = train_one_fold(args, fold_id, device)
        fold_results.append(fold_result)

    summary, overall = summarize_cv_results(fold_results, args.num_classes)

    print(f"\n{'=' * 90}")
    print("五折交叉验证汇总结果（按每折指标求 mean ± std）")
    print(f"Accuracy : {summary['acc']['mean']:.4f} ± {summary['acc']['std']:.4f}")
    print(f"Precision: {summary['precision_macro']['mean']:.4f} ± {summary['precision_macro']['std']:.4f}")
    print(f"Recall   : {summary['recall_macro']['mean']:.4f} ± {summary['recall_macro']['std']:.4f}")
    print(f"F1-score : {summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}")

    print("\n把五个验证折的预测拼接后得到的总体指标")
    print(f"Overall Accuracy : {overall['acc']:.4f}")
    print(f"Overall Precision: {overall['precision_macro']:.4f}")
    print(f"Overall Recall   : {overall['recall_macro']:.4f}")
    print(f"Overall F1-score : {overall['f1_macro']:.4f}")
    print(f"总体样本数         : {overall['num_samples']}")
    print(f"{'=' * 90}")

    save_obj = {
        "folder_id": tag,
        "device": str(device),
        "config": vars(args),
        "summary_mean_std": summary,
        "overall_oof_metrics": overall,
        "fold_results": fold_results,
    }

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {args.result_json}")


if __name__ == "__main__":
    main()
