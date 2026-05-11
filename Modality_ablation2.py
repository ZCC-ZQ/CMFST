import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


# 你的文件名(驼峰) -> 模型pool_mode(下划线) 的映射
MODE_SPECS = {
    "mean":       {"pool_mode": "mean",        "fname": "full_mean{tag}_seed{seed}.pt"},
    "max":        {"pool_mode": "max",         "fname": "full_max{tag}_seed{seed}.pt"},
    "last":       {"pool_mode": "last",        "fname": "full_last{tag}_seed{seed}.pt"},
    "selfattn":   {"pool_mode": "selfattn",    "fname": "full_selfattn{tag}_seed{seed}.pt"},
    "gate":       {"pool_mode": "gate",        "fname": "full_gate{tag}_seed{seed}.pt"},
    "gateGlobal": {"pool_mode": "gate_global", "fname": "full_gateGlobal{tag}_seed{seed}.pt"},
    "gateLocal":  {"pool_mode": "gate_local",  "fname": "full_gateLocal{tag}_seed{seed}.pt"},
}


@torch.no_grad()
def eval_one_ckpt(model, loader, device, ckpt_path, pool_mode, ablation="full", avg="macro"):
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

        out = model(audio, local, global_t, mask, ablation=ablation, pool_mode=pool_mode)
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
    x = np.array(arr_2d, dtype=float)  # (N,4)
    return x.mean(axis=0), x.std(axis=0)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--test_txt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    # ckpt
    parser.add_argument("--ckpt_dir", type=str, default="ckpts")
    parser.add_argument("--seeds", type=str, default="38,39,40,41,42")
    parser.add_argument("--tag", type=str, default="", help="e.g. en -> full_xxx_en_seed{seed}.pt")
    parser.add_argument(
        "--modes",
        type=str,
        default="mean,max,last,selfattn,gate,gateGlobal,gateLocal",
        help="comma-separated: mean,max,last,selfattn,gate,gateGlobal,gateLocal"
    )

    # model hyperparams (必须与训练一致)
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=96)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_factors", type=int, default=12)

    # ablation固定full做Ⅱ类
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "audio", "text"])
    parser.add_argument("--avg", type=str, default="macro", choices=["macro", "weighted"])
    parser.add_argument("--print_each_ckpt", action="store_true")

    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    for m in modes:
        if m not in MODE_SPECS:
            raise ValueError(f"Unknown mode='{m}'. Valid: {list(MODE_SPECS.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_set = GateDataset(args.test_txt)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # model
    model = GateAggWithBiGRUClassifier(
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        rnn_hidden=args.rnn_hidden,
        hidden_gate=args.hidden_gate,
        hidden_main=args.hidden_main,
        num_main_classes=args.num_classes,
        num_factors=args.num_factors,
    ).to(device)

    # eval each mode
    for mode in modes:
        spec = MODE_SPECS[mode]
        pool_mode = spec["pool_mode"]
        fname_tpl = spec["fname"]

        tag = f"_{args.tag}" if args.tag else ""
        ckpts = [
            os.path.join(args.ckpt_dir, fname_tpl.format(seed=s, tag=tag))
            for s in seeds
        ]
        for p in ckpts:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[{mode}] missing ckpt: {p}")

        metrics = []
        for p in ckpts:
            acc, pre, rec, f1 = eval_one_ckpt(
                model, test_loader, device,
                ckpt_path=p, pool_mode=pool_mode,
                ablation=args.ablation, avg=args.avg
            )
            metrics.append([acc, pre, rec, f1])
            if args.print_each_ckpt:
                bn = os.path.basename(p)
                print(f"[{mode}] {bn} | Acc={acc:.4f} Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f}")

        mean, std = mean_std(metrics)
        print(
            f"{mode} (pool_mode={pool_mode}, avg={args.avg}): "
            f"Acc {mean[0]:.4f}±{std[0]:.4f}, "
            f"Pre {mean[1]:.4f}±{std[1]:.4f}, "
            f"Rec {mean[2]:.4f}±{std[2]:.4f}, "
            f"F1 {mean[3]:.4f}±{std[3]:.4f}"
        )


if __name__ == "__main__":
    main()
