import os
import argparse
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

    avg_loss = total_loss / total_n
    avg_acc = total_correct / total_n

    all_logits = torch.cat(all_logits, dim=0)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    return avg_loss, avg_acc, all_logits, all_labels


def plot_confusion_matrix_sci(
    cm,
    class_names,
    save_path="confmat.pdf",
    normalize="true",     # "true" (row) | "pred" (col) | None (count)
    cmap="Blues",
    dpi=600
):
    """
    SCI风格混淆矩阵绘图（matplotlib，无seaborn）
    - normalize="true": 行归一化（每个真实类的百分比，最常用）
    - normalize="pred": 列归一化
    - normalize=None: 原始计数
    """

    # ---------- 全局风格：Times New Roman + 8pt + 统一线宽 ----------
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })

    cm = np.asarray(cm, dtype=float)
    cm_count = cm.copy()

    # ---------- 归一化 ----------
    if normalize == "true":
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)
        vmax = 1.0
    elif normalize == "pred":
        col_sum = cm.sum(axis=0, keepdims=True)
        cm_show = np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum != 0)
        vmax = 1.0
    elif normalize is None:
        cm_show = cm_count
        vmax = None
    else:
        raise ValueError("normalize must be one of: 'true', 'pred', None")

    n = cm_show.shape[0]

    # 论文常用尺寸（单栏/半栏），可按版面再微调
    fig, ax = plt.subplots(figsize=(3.25, 3.0))

    im = ax.imshow(cm_show, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)

    # ---------- colorbar ----------
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3.5)
    cbar.outline.set_linewidth(0.8)

    # ---------- 轴标签/刻度 ----------
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=4)
    ax.set_ylabel("True", fontsize=8, labelpad=4)
    ax.set_aspect("equal")

    # ---------- 细白色分隔线（提升可读性，线不要太粗） ----------
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---------- 边框线宽 ----------
    for s in ["left", "bottom", "right", "top"]:
        ax.spines[s].set_linewidth(0.8)

    # ---------- 数值标注：count + 百分比（归一化时） ----------
    # 根据底色自动切换字体颜色
    # ---------- 数值标注：只显示 count ----------
    thresh = (cm_show.max() if cm_show.size else 0) * 0.55
    for i in range(n):
        for j in range(n):
            txt = f"{int(cm_count[i, j])}"
            val = cm_show[i, j]  # 颜色阈值仍按归一化后的深浅判断（更合理）

            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8,
                color="white" if val > thresh else "black"
            )

    fig.tight_layout(pad=0.3)

    # ---------- 输出：PDF + PNG ----------
    base, ext = os.path.splitext(save_path)
    if ext.lower() == ".pdf":
        out_pdf = save_path
        out_png = base + ".png"
    elif ext.lower() == ".png":
        out_png = save_path
        out_pdf = base + ".pdf"
    else:
        out_pdf = base + ".pdf"
        out_png = base + ".png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_pdf}")
    print(f"[Saved] {out_png}")

@torch.no_grad()
def eval_ensemble(ckpt_paths, test_txt,
                  num_classes=2,
                  rnn_hidden=256,
                  hidden_gate=256,
                  hidden_main=96,
                  num_factors=8):
    """
    对若干个 ckpt 做集成评估
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 构建测试集
    test_set = GateDataset(test_txt)
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 2) 逐个模型评估
    logits_list = []
    labels_ref = None

    for ckpt in ckpt_paths:
        print(f"Evaluating single model: {ckpt}")

        model = GateAggWithBiGRUClassifier(
            audio_dim=768,
            text_dim=768,
            rnn_hidden=rnn_hidden,
            hidden_gate=hidden_gate,
            hidden_main=hidden_main,
            num_main_classes=num_classes,
            num_factors=num_factors  # [修改] 使用传入的变量，而不是写死 12
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        loss, acc, all_logits, all_labels = eval_single_model(model, test_loader, device)
        print(f"  single_model_loss={loss:.4f}, single_model_acc={acc:.4f}")

        logits_list.append(all_logits)
        if labels_ref is None:
            labels_ref = all_labels
        else:
            # 保险起见，可以断言一下标签一致
            assert torch.equal(labels_ref, all_labels), "不同 ckpt 评估得到的标签顺序不一致！"

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
    print(f"  ensemble_acc = {acc_ens:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    # 你的标签编码：0-AD / 1-HC / 2-MCI
    class_names = ["AD", "HC"]


    plot_confusion_matrix_sci(
        cm,
        class_names=class_names,
        save_path="confmat_AD_HC_en.pdf",
        normalize="true",  # 行归一化（推荐）
        cmap="Blues",
        dpi=600
    )

    # ================= [新增] 计算敏感度 (Sensitivity) 和 特异度 (Specificity) =================
    print("\n" + "=" * 20 + " Detailed Sensitivity & Specificity " + "=" * 20)
    sens_list = []
    spec_list = []

    for c in range(num_classes):
        # TP: 该类被正确预测为该类的数量 (对角线)
        tp = cm[c, c]

        # FN: 该类被预测为其他类的数量 (该行之和 - TP)
        fn = cm[c, :].sum() - tp

        # FP: 其他类被预测为该类的数量 (该列之和 - TP)
        fp = cm[:, c].sum() - tp

        # TN: 其他类被正确预测为其他类的数量 (总数 - 该行 - 该列 + TP)
        tn = cm.sum() - (tp + fn + fp)

        # 计算敏感度 (Sensitivity) = Recall = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 计算特异度 (Specificity) = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sens_list.append(sensitivity)
        spec_list.append(specificity)

        print(f"  Class {c} (vs Rest):")
        print(f"    Sensitivity (Recall) : {sensitivity:.4f}")
        print(f"    Specificity          : {specificity:.4f}")

    # 计算宏平均 (Macro Average)
    macro_sens = sum(sens_list) / num_classes
    macro_spec = sum(spec_list) / num_classes

    print("-" * 60)
    print(f"  Macro Sensitivity      : {macro_sens:.4f}")
    print(f"  Macro Specificity      : {macro_spec:.4f}")
    print("=" * 76 + "\n")
    # =========================================================================================

    # per-class precision/recall/F1
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    print("\nPer-class metrics:")
    for c in range(num_classes):
        print(f"  Class {c}: precision={prec[c]:.4f}, recall={rec[c]:.4f}, "
              f"f1={f1[c]:.4f}, support={support[c]}")

    # macro / weighted
    prec_macro = prec.mean()
    rec_macro = rec.mean()
    f1_macro = f1.mean()

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average="weighted", zero_division=0
    )

    print("\nMacro-averaged:")
    print(f"  precision={prec_macro:.4f}, recall={rec_macro:.4f}, f1={f1_macro:.4f}")

    print("\nWeighted-averaged:")
    print(f"  precision={prec_w:.4f}, recall={rec_w:.4f}, f1={f1_w:.4f}")

    # sklearn 自带的详细报告
    target_names = [f"class_{i}" for i in range(num_classes)]
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=target_names, digits=4, zero_division=0
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt", type=str,
                        default="dataset/list_ADReSSo_fusion/test.txt")
    # 注意：ckpts 路径记得改成你实际要评估的文件
    parser.add_argument("--ckpts", type=str, nargs="+",
                        default=[
                            "best_model/best_gate_adresso_cls_seeed2.pt"
                        ])
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)

    # [注意] 这里 default 改为 96 比较安全，因为 model.py 默认是 96
    # 如果你训练时用了 156，运行时记得用 --hidden_main 156 指定
    parser.add_argument("--hidden_main", type=int, default=96)

    parser.add_argument("--num_classes", type=int, default=2)

    # [新增] 添加 num_factors 参数
    parser.add_argument("--num_factors", type=int, default=8)

    args = parser.parse_args()

    # [修改] 调用函数时传入 num_factors
    eval_ensemble(
        args.ckpts,
        args.test_txt,
        num_classes=args.num_classes,
        rnn_hidden=args.rnn_hidden,
        hidden_gate=args.hidden_gate,
        hidden_main=args.hidden_main,
        num_factors=args.num_factors
    )