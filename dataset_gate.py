import numpy as np
import torch
from torch.utils.data import Dataset

class GateDataset(Dataset):
    def __init__(self, list_txt):
        """
        list_txt 每行格式：
        audio_path text_embeds_path global_text_path label_int
        （三列路径 + 一列标签）
        """
        self.items = []
        with open(list_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"行列数不是 4: {line}")
                a_path, lt_path, g_path, label = parts
                self.items.append((a_path, lt_path, g_path, int(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a_path, lt_path, g_path, label = self.items[idx]

        audio_embeds = np.load(a_path)          # (T,768)
        local_text_embeds = np.load(lt_path)    # (T,768)
        global_text_embed = np.load(g_path)     # (768,) or (1,768)

        audio_embeds = torch.from_numpy(audio_embeds).float()
        local_text_embeds = torch.from_numpy(local_text_embeds).float()
        global_text_embed = torch.from_numpy(global_text_embed).float()
        if global_text_embed.ndim == 2:  # (1,768) -> (768,)
            global_text_embed = global_text_embed.squeeze(0)

        T = audio_embeds.shape[0]
        window_mask = torch.ones(T, dtype=torch.float32)

        return {
            "audio_embeds": audio_embeds,
            "local_text_embeds": local_text_embeds,
            "global_text_embed": global_text_embed,
            "window_mask": window_mask,
            "y_main": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """按当前 batch 内最长 T padding。"""
    B = len(batch)
    T_max = max(x["audio_embeds"].shape[0] for x in batch)
    D_a = batch[0]["audio_embeds"].shape[1]
    D_t = batch[0]["local_text_embeds"].shape[1]

    audio_batch, local_batch = [], []
    global_batch, mask_batch, y_batch = [], [], []

    for x in batch:
        a = x["audio_embeds"]
        lt = x["local_text_embeds"]
        g = x["global_text_embed"]
        m = x["window_mask"]
        y = x["y_main"]

        T = a.shape[0]
        pad = T_max - T
        if pad > 0:
            a = torch.cat([a, torch.zeros(pad, D_a)], dim=0)
            lt = torch.cat([lt, torch.zeros(pad, D_t)], dim=0)
            m = torch.cat([m, torch.zeros(pad)], dim=0)

        audio_batch.append(a)
        local_batch.append(lt)
        global_batch.append(g)
        mask_batch.append(m)
        y_batch.append(y)

    return {
        "audio_embeds": torch.stack(audio_batch, 0),        # (B,T_max,768)
        "local_text_embeds": torch.stack(local_batch, 0),   # (B,T_max,768)
        "global_text_embeds": torch.stack(global_batch, 0), # (B,768)
        "window_mask": torch.stack(mask_batch, 0),          # (B,T_max)
        "y_main": torch.stack(y_batch, 0),                  # (B,)
    }
