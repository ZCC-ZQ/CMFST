import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGlobalGateAttentionPooling(nn.Module):
    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=256):
        super().__init__()
        # [e_i || t_i || t_global] -> gate g_i
        self.gate_fc = nn.Sequential(
            nn.Linear(audio_dim + text_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 只基于音频的 attention 打分 u_i
        self.att_fc = nn.Linear(audio_dim, 1)

    def forward(self, audio_embeds, local_text_embeds, global_text_embeds, window_mask=None):
        """
        audio_embeds:      (B, T, D_a)
        local_text_embeds: (B, T, D_t)
        global_text_embeds:(B, D_t)
        window_mask:       (B, T)  1=有效, 0=无效
        """
        B, T, D_a = audio_embeds.shape
        _, _, D_t = local_text_embeds.shape

        # (B, D_t) -> (B, 1, D_t) -> (B, T, D_t)
        global_expand = global_text_embeds.unsqueeze(1).expand(-1, T, -1)

        # 拼接 [e_i || t_i || t_global] 作为 gate 输入
        gate_input = torch.cat([audio_embeds, local_text_embeds, global_expand], dim=-1)  # (B,T,D_a+2*D_t)

        # 门控 g_i ∈ (0,1)
        gate_g = torch.sigmoid(self.gate_fc(gate_input))  # (B,T,1)

        # 音频 attention 打分 u_i
        u = self.att_fc(audio_embeds)  # (B,T,1)

        # 综合打分 scores_i = g_i * u_i
        scores = gate_g * u  # (B,T,1)

        # 屏蔽 padding 窗口
        if window_mask is not None:
            mask = window_mask.unsqueeze(-1).float()  # (B,T,1)
            scores = scores + (1.0 - mask) * (-1e9)

        # 注意力权重 α_i
        alpha = F.softmax(scores, dim=1)  # (B,T,1)

        # 加权求和 -> 整段 embedding
        pooled = torch.sum(alpha * audio_embeds, dim=1)  # (B,D_a)

        return pooled, alpha, gate_g
