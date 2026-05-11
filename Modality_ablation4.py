import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


MODULE_CONFIG = {
    "base": {
        "ckpt_dir": "ckpts/module_base",
        "prefix": "base",
        "module_ablation": "base",
    },
    "lgsga": {
        "ckpt_dir": "ckpts/module_lgsga",
        "prefix": "lgsga",
        "module_ablation": "lgsga",
    },
    "atcr": {
        "ckpt_dir": "ckpts/module_atcr",
        "prefix": "atcr",
        "module_ablation": "atcr",
    },
    "full": {
        "ckpt_dir": "ckpts/module_full",
        "prefix": "full",
        "module_ablation": "full",
    },
}


@torch.no_grad()
def eval_one_ckpt(model, loader, device, ckpt_path,
                  ablation="full", module_ablation="full", avg="macro"):
    state = torch.load(ckpt_path, map_location=device)

    # 兼容两种保存方式
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    y_true, y_pred = [], []

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

    # 一次性评估哪些模块，默认就是你要的 3 个
    parser.add_argument(
        "--modules",
        type=str,
        default="base,lgsga,atcr",
        help="comma-separated modules, e.g. base,lgsga,atcr or base,lgsga,atcr,full"
    )

    # 第几次运行，对应 base_1.pt / base_2.pt ...
    parser.add_argument("--runs", type=str, default="1,2,3,4,5")

    # 模态消融保持 full，避免和模块消融混用
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "audio", "text"]
    )

    parser.add_argument(
        "--avg",
        type=str,
        default="macro",
        choices=["macro", "weighted"]
    )

    # model hyperparams
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--hidden_gate", type=int, default=256)
    parser.add_argument("--hidden_main", type=int, default=96)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_factors", type=int, default=12)

    parser.add_argument("--print_each_ckpt", action="store_true")

    args = parser.parse_args()

    modules = [x.strip() for x in args.modules.split(",") if x.strip()]
    runs = [int(x.strip()) for x in args.runs.split(",") if x.strip()]

    for m in modules:
        if m not in MODULE_CONFIG:
            raise ValueError(f"Unknown module '{m}'. Valid: {list(MODULE_CONFIG.keys())}")

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

    print("=" * 100)
    print(f"test_txt={args.test_txt}")
    print(f"modules={modules}")
    print(f"runs={runs}")
    print(f"ablation={args.ablation}")
    print(f"avg={args.avg}")
    print("=" * 100)

    all_results = {}

    for m in modules:
        cfg = MODULE_CONFIG[m]
        ckpt_dir = cfg["ckpt_dir"]
        prefix = cfg["prefix"]
        module_ablation = cfg["module_ablation"]

        ckpts = [os.path.join(ckpt_dir, f"{prefix}_{r}.pt") for r in runs]

        for p in ckpts:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[{m}] missing ckpt: {p}")

        metrics = []
        for p in ckpts:
            acc, pre, rec, f1 = eval_one_ckpt(
                model=model,
                loader=test_loader,
                device=device,
                ckpt_path=p,
                ablation=args.ablation,
                module_ablation=module_ablation,
                avg=args.avg,
            )
            metrics.append([acc, pre, rec, f1])

            if args.print_each_ckpt:
                print(
                    f"[{m}] {os.path.basename(p)} | "
                    f"Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}"
                )

        mean, std = mean_std(metrics)
        all_results[m] = (mean, std)

    print("Summary:")
    for m in modules:
        mean, std = all_results[m]
        print(
            f"{m:<6} | "
            f"Acc {mean[0]:.4f}±{std[0]:.4f} | "
            f"Pre {mean[1]:.4f}±{std[1]:.4f} | "
            f"Rec {mean[2]:.4f}±{std[2]:.4f} | "
            f"F1 {mean[3]:.4f}±{std[3]:.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()