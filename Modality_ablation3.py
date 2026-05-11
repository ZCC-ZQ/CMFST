import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


BASE_FOLDERS = [
    "factor_on",
    "factor_off",
    "factor_audio_only",
    "factor_text_only",
    "factor_no_gate",
    "factor_no_scale",
]

FOLDER2FMODE = {
    "factor_on": "on",
    "factor_off": "off",
    "factor_audio_only": "audio_only",
    "factor_text_only": "text_only",
    "factor_no_gate": "no_gate",
    "factor_no_scale": "no_scale",
}


@torch.no_grad()
def eval_one_ckpt(model, loader, device, ckpt_path,
                 ablation="full", pool_mode="gate", factor_mode="on", avg="macro"):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    y_true, y_pred = [], []
    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"].to(device)

        out = model(audio, local, global_t, mask,
                    ablation=ablation, pool_mode=pool_mode, factor_mode=factor_mode)
        pred = out["logits"].argmax(dim=1)

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0
    )
    return acc, pre, rec, f1


def mean_std(arr_2d):
    x = np.array(arr_2d, dtype=float)
    return x.mean(axis=0), x.std(axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_txt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--ckpt_root", type=str, default="ckpts")
    parser.add_argument("--seeds", type=str, default="38,39,40,41,42")

    # 关键：英文模型在文件夹后缀 _en
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])

    # 可选：只评估部分分支
    parser.add_argument(
        "--folders", type=str, default="",
        help="comma-separated base folders, e.g. factor_on,factor_off. empty=all"
    )

    parser.add_argument("--ablation", type=str, default="full", choices=["full", "audio", "text"])
    parser.add_argument(
        "--pooling", type=str, default="gate",
        choices=["mean", "max", "last", "selfattn", "gate", "gate_global", "gate_local"]
    )

    # model hyperparams
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=96)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_factors", type=int, default=12)

    parser.add_argument("--avg", type=str, default="macro", choices=["macro", "weighted"])
    parser.add_argument("--print_each_ckpt", action="store_true")

    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    if args.folders.strip():
        base_folders = [x.strip() for x in args.folders.split(",") if x.strip()]
    else:
        base_folders = BASE_FOLDERS

    for bf in base_folders:
        if bf not in FOLDER2FMODE:
            raise ValueError(f"Unknown folder='{bf}'. Valid: {list(FOLDER2FMODE.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = GateDataset(args.test_txt)
    test_loader = DataLoader(
        test_set,
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

    # 文件名固定不带 _en
    fname_tpl = "full_gate_seed{seed}.pt"

    for bf in base_folders:
        folder = bf + ("_en" if args.lang == "en" else "")
        fmode = FOLDER2FMODE[bf]

        ckpt_dir = os.path.join(args.ckpt_root, folder)
        ckpts = [os.path.join(ckpt_dir, fname_tpl.format(seed=s)) for s in seeds]

        for p in ckpts:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[{folder}] missing ckpt: {p}")

        metrics = []
        for p in ckpts:
            acc, pre, rec, f1 = eval_one_ckpt(
                model, test_loader, device,
                ckpt_path=p,
                ablation=args.ablation,
                pool_mode=args.pooling,
                factor_mode=fmode,
                avg=args.avg
            )
            metrics.append([acc, pre, rec, f1])
            print(f"[{folder}/{fmode}] {os.path.basename(p)} | Acc={acc:.4f} Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f}")

        mean, std = mean_std(metrics)
        print(
            f"{folder} (factor_mode={fmode}, pool={args.pooling}, avg={args.avg}): "
            f"Acc {mean[0]:.4f}±{std[0]:.4f}, "
            f"Pre {mean[1]:.4f}±{std[1]:.4f}, "
            f"Rec {mean[2]:.4f}±{std[2]:.4f}, "
            f"F1 {mean[3]:.4f}±{std[3]:.4f}"
        )


if __name__ == "__main__":
    main()
