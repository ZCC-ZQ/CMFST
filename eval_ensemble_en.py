# eval_ensemble_binary.py
import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)


@torch.no_grad()
def eval_single_model(model, loader, device):
    """
    对单个模型在给定 loader 上做评估，返回：
      - avg_loss, avg_acc
      - all_logits: (N, C) tensor
      - all_labels: (N,) tensor
    """
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    total_loss, total_correct, total_n = 0.0, 0, 0
    all_logits = []
    all_labels = []

    for batch in loader:
        audio = batch["audio_embeds"].to(device)          # (B, T, 768)
        local = batch["local_text_embeds"].to(device)     # (B, T, 768)
        global_t = batch["global_text_embeds"].to(device) # (B, 768)
        mask = batch["window_mask"].to(device)            # (B, T)
        y = batch["y_main"].to(device)                    # (B,)

        out = model(audio, local, global_t, mask)
        logits = out["logits"]                            # (B, C)

        loss = ce(logits, y)

        pred = logits.argmax(dim=-1)
        correct = (pred == y).sum().item()
        n = y.size(0)

        total_loss += loss.item() * n
        total_correct += correct
        total_n += n

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / max(total_n, 1)
    avg_acc = total_correct / max(total_n, 1)

    all_logits = torch.cat(all_logits, dim=0)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    return avg_loss, avg_acc, all_logits, all_labels


@torch.no_grad()
def eval_ensemble(
    ckpt_paths,
    test_txt,
    num_classes=2,
    rnn_hidden=256,
    hidden_gate=256,
    hidden_main=96,
    num_factors=12,
    batch_size=8,
):
    """
    对若干个 ckpt 做二分类集成评估：
      - 先分别跑单模型评估
      - 再对 logits 做平均，算 ensemble 的指标
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 构建测试集（英文 ADReSSo）
    test_set = GateDataset(test_txt)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 2) 逐个模型评估
    logits_list = []
    labels_ref = None

    for ckpt in ckpt_paths:
        print(f"\nEvaluating single model: {ckpt}")
        model = GateAggWithBiGRUClassifier(
            audio_dim=768,
            text_dim=768,
            rnn_hidden=rnn_hidden,   # 必须和训练该 ckpt 时保持一致
            hidden_gate=hidden_gate,
            hidden_main=hidden_main,
            num_main_classes=num_classes,
            num_factors=num_factors,
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        loss, acc, all_logits, all_labels = eval_single_model(model, test_loader, device)
        print(f"  single_model_loss={loss:.4f}, single_model_acc={acc:.4f}")

        logits_list.append(all_logits)
        if labels_ref is None:
            labels_ref = all_labels
        else:
            # 保险起见，断言标签顺序一致
            assert torch.equal(labels_ref, all_labels), (
                "不同 ckpt 评估得到的标签顺序不一致，请确认 test_txt 和 DataLoader 完全相同！"
            )

    # 3) logits 集成（简单平均）
    if len(logits_list) == 1:
        print("\nOnly one model given, ensemble == single model.")
        ensemble_logits = logits_list[0]
    else:
        print(f"\nEnsemble over {len(logits_list)} models")
        stacked_logits = torch.stack(logits_list, dim=0)  # (M, N, C)
        ensemble_logits = stacked_logits.mean(dim=0)      # (N, C)

    # 4) 计算各种指标
    y_true = labels_ref.numpy()
    y_pred = ensemble_logits.argmax(dim=-1).numpy()

    acc_ens = (y_true == y_pred).mean()
    print(f"\nEnsemble accuracy = {acc_ens:.4f}")

    # 混淆矩阵（行 = 真值，列 = 预测）
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    # per-class precision / recall / F1
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    print("\nPer-class metrics:")
    for c in range(num_classes):
        print(
            f"  Class {c}: precision={prec[c]:.4f}, "
            f"recall={rec[c]:.4f}, f1={f1[c]:.4f}, support={support[c]}"
        )

    # macro / weighted
    prec_macro = prec.mean()
    rec_macro = rec.mean()
    f1_macro = f1.mean()

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="weighted",
        zero_division=0,
    )

    print("\nMacro-averaged:")
    print(
        f"  precision={prec_macro:.4f}, "
        f"recall={rec_macro:.4f}, f1={f1_macro:.4f}"
    )

    print("\nWeighted-averaged:")
    print(
        f"  precision={prec_w:.4f}, "
        f"recall={rec_w:.4f}, f1={f1_w:.4f}"
    )

    # sklearn 自带的详细报告
    target_names = [f"class_{i}" for i in range(num_classes)]
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_txt",
        type=str,
        default="dataset/list_ADReSSo_fusion/test.txt",  # 英文二分类 list
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        default=[
                "best_modelmodel/best_gate_adresso_cls_seed3.pt",
        ],
    )

    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=128)
    parser.add_argument("--num_factors", type=int, default=8)

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    eval_ensemble(
        ckpt_paths=args.ckpts,
        test_txt=args.test_txt,
        num_classes=args.num_classes,
        rnn_hidden=args.rnn_hidden,
        hidden_gate=args.hidden_gate,
        hidden_main=args.hidden_main,
        num_factors=args.num_factors,
        batch_size=args.batch_size,
    )
