import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier


@torch.no_grad()
def eval_one_ckpt(model, loader, device, ckpt_path: str, ablation: str):
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

        out = model(audio, local, global_t, mask, ablation=ablation)
        pred = out["logits"].argmax(dim=1)

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, pre, rec, f1

def _infer_seed_from_path(p: str):
    # 兼容 ...seed38.pt / ...seed_38.pt / ...seed-38.pt
    m = re.search(r"seed[_-]?(\d+)", os.path.basename(p), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def eval_mode(mode: str, ckpts, model, loader, device, verbose=True):
    if len(ckpts) != 5:
        raise ValueError(f"[{mode}] 需要 5 个权重，但给了 {len(ckpts)} 个：{ckpts}")
    for p in ckpts:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[{mode}] 找不到权重文件：{p}")

    rows = []  # 每个seed一行：seed, acc, pre, rec, f1
    for p in ckpts:
        acc, pre, rec, f1 = eval_one_ckpt(model, loader, device, p, ablation=mode)

        seed = _infer_seed_from_path(p)
        rows.append([seed, acc, pre, rec, f1])

        if verbose:
            seed_str = f"{seed}" if seed is not None else os.path.basename(p)
            print(
                f"[{mode}] seed={seed_str}: "
                f"Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
            )

    rows = np.array(rows, dtype=object)  # (5,5)
    metrics = rows[:, 1:].astype(float)  # (5,4) acc/pre/rec/f1
    mean = metrics.mean(axis=0)
    std = metrics.std(axis=0)

    return mean, std, rows


def main():
    # ===================== 1) 直接在代码里写死路径（你只改这一段） =====================
    test_txt_path = r"dataset/list_NCMMSC2021/test.txt"   # 测试集list路径

    ckpt_paths = {
        "audio": [
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\audio_seed38.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\audio_seed39.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\audio_seed40.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\audio_seed41.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\audio_seed42.pt",
        ],
        "text": [
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\text_seed38.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\text_seed39.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\text_seed40.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\text_seed41.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\text_seed42.pt",
        ],
        "full": [
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\full_seed38.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\full_seed39.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\full_seed40.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\full_seed41.pt",
            r"C:\Users\smart332\Desktop\gate_attention(best)\ckpts\full_seed42.pt",
        ],
    }
    # ================================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = GateDataset(test_txt_path)
    test_loader = DataLoader(
        test_set, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # 模型结构参数必须与你训练时一致
    model = GateAggWithBiGRUClassifier(
        audio_dim=768,
        text_dim=768,
        rnn_hidden=256,
        hidden_gate=256,
        hidden_main=96,
        num_main_classes=3,
        num_factors=12,
    ).to(device)

    for mode in ["audio", "text", "full"]:
        mean, std, rows = eval_mode(mode, ckpt_paths[mode], model, test_loader, device, verbose=True)
        print(
            f"{mode}: "
            f"Acc {mean[0]:.4f}±{std[0]:.4f}, "
            f"Pre {mean[1]:.4f}±{std[1]:.4f}, "
            f"Rec {mean[2]:.4f}±{std[2]:.4f}, "
            f"F1 {mean[3]:.4f}±{std[3]:.4f}"
        )


if __name__ == "__main__":
    main()
