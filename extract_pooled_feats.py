import torch
from torch.utils.data import DataLoader
from dataset_gate import GateDataset, collate_fn
from model_gate_cls import GateAggWithClassifier

@torch.no_grad()
def extract(list_txt, ckpt_path, out_path, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GateDataset(list_txt)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = GateAggWithClassifier(
        audio_dim=768,
        text_dim=768,
        num_main_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_feats = []
    all_labels = []

    for batch in loader:
        audio = batch["audio_embeds"].to(device)
        local = batch["local_text_embeds"].to(device)
        global_t = batch["global_text_embeds"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y_main"]  # 保留在 CPU 即可

        out = model(audio, local, global_t, mask)
        pooled = out["pooled"].cpu()   # (B,768)

        all_feats.append(pooled)
        all_labels.append(y)

    all_feats = torch.cat(all_feats, dim=0)   # (N,768)
    all_labels = torch.cat(all_labels, dim=0) # (N,)

    torch.save(
        {"feats": all_feats, "labels": all_labels},
        out_path
    )
    print(f"saved pooled feats to {out_path}, feats.shape={all_feats.shape}, labels.shape={all_labels.shape}")


if __name__ == "__main__":
    # 先确保已经训练好 best_gate_cls.pt
    extract(
        list_txt="dataset/list_NCMMSC2021_AT/train.txt",
        ckpt_path="best_gate_cls.pt",
        out_path="train_pooled.pt",
        num_classes=3
    )
    extract(
        list_txt="dataset/list_NCMMSC2021_AT/test.txt",
        ckpt_path="best_gate_cls.pt",
        out_path="test_pooled.pt",
        num_classes=3
    )
