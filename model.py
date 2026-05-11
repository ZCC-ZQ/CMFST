import torch
import torch.nn as nn
from gate_pool import LocalGlobalGateAttentionPooling


class GateAggWithBiGRUClassifier(nn.Module):
    def __init__(
        self,
        audio_dim: int = 768,
        text_dim: int = 768,
        rnn_hidden: int = 256,
        hidden_gate: int = 256,
        hidden_main: int = 96,
        num_main_classes: int = 2,
        num_factors: int = 8,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_factors = num_factors

        # ===== 1) 音频上下文编码 =====
        self.rnn = nn.GRU(
            input_size=audio_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_proj = nn.Linear(rnn_hidden * 2, audio_dim)

        # ===== 2) LGSGA：局部-全局语义引导的音频聚合 =====
        self.gate_pool = LocalGlobalGateAttentionPooling(
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_gate,
        )

        # ===== 3) ATCR：音频-文本协同表征分支 =====
        self.W_a = nn.Linear(audio_dim, num_factors)
        self.W_t = nn.Linear(text_dim, num_factors)
        self.gate_a_net = nn.Linear(audio_dim + text_dim, num_factors)
        self.gate_t_net = nn.Linear(audio_dim + text_dim, num_factors)

        # ===== 4) 主干融合 =====
        fusion_in_dim = audio_dim + text_dim  # 768 + 768 = 1536
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, audio_dim),  # 1536 -> 768
            nn.LayerNorm(audio_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ===== 5) 分类头 =====
        self.head_hidden = nn.Sequential(
            nn.Linear(audio_dim, hidden_main),  # 768 -> 96
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # 为了便于公平比较，分类头始终保持 hidden_main + num_factors 输入维度不变
        final_in_dim = hidden_main + num_factors
        self.head_final = nn.Linear(final_in_dim, num_main_classes)

    def _masked_mean_pool(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [B, T]，有效位置为1，无效位置为0
        """
        if mask is None:
            return x.mean(dim=1)

        m = mask.float().unsqueeze(-1)  # [B, T, 1]
        denom = m.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        pooled = (x * m).sum(dim=1) / denom
        return pooled

    def forward(
        self,
        audio_embeds,
        local_text_embeds,
        global_text_embeds,
        window_mask=None,
        ablation: str = "full",           # 兼容旧代码：full | audio | text
        module_ablation: str = "full",    # 新增：base | lgsga | atcr | full
    ):
        """
        ablation:
            - full  : 音频+文本都用（默认）
            - audio : 仅音频模态（文本置零）
            - text  : 仅文本模态（音频置零）

        module_ablation:
            - base  : 不用 LGSGA，不用 ATCR
            - lgsga : 只用 LGSGA，不用 ATCR
            - atcr  : 不用 LGSGA，只用 ATCR
            - full  : LGSGA + ATCR 全部使用
        """
        ablation = (ablation or "full").lower()
        module_ablation = (module_ablation or "full").lower()

        if ablation not in {"full", "audio", "text"}:
            raise ValueError(f"Unknown ablation='{ablation}'. Use one of: full/audio/text")

        if module_ablation not in {"base", "lgsga", "atcr", "full"}:
            raise ValueError(
                f"Unknown module_ablation='{module_ablation}'. "
                f"Use one of: base/lgsga/atcr/full"
            )

        B = global_text_embeds.size(0)
        device = global_text_embeds.device
        dtype = global_text_embeds.dtype

        # -------------------------------------------------
        # A. 先处理模态消融（兼容你原来的 audio/text/full）
        # -------------------------------------------------
        if ablation == "audio":
            local_text_for_pool = torch.zeros_like(local_text_embeds)
            global_text_for_pool = torch.zeros_like(global_text_embeds)
            global_text_for_fusion = torch.zeros_like(global_text_embeds)
            global_text_for_factor = torch.zeros_like(global_text_embeds)
        else:
            local_text_for_pool = local_text_embeds
            global_text_for_pool = global_text_embeds
            global_text_for_fusion = global_text_embeds
            global_text_for_factor = global_text_embeds

        # -------------------------------------------------
        # B. 音频分支：先 BiGRU，再决定是否使用 LGSGA
        # -------------------------------------------------
        if ablation == "text":
            # 文本-only：音频分支直接置零
            pooled_audio = torch.zeros(B, self.audio_dim, device=device, dtype=dtype)
            alpha = None
            gate_pool_g = None
        else:
            rnn_out, _ = self.rnn(audio_embeds)      # [B, T, 2H]
            audio_ctx = self.rnn_proj(rnn_out)       # [B, T, 768]

            if module_ablation in {"lgsga", "full"}:
                # 使用 LGSGA
                pooled_audio, alpha, gate_pool_g = self.gate_pool(
                    audio_ctx,
                    local_text_for_pool,
                    global_text_for_pool,
                    window_mask,
                )
            else:
                # base / atcr：不用 LGSGA，改为普通 masked mean pooling
                pooled_audio = self._masked_mean_pool(audio_ctx, window_mask)
                alpha = None
                gate_pool_g = None

        # -------------------------------------------------
        # C. ATCR 协同表征分支
        # -------------------------------------------------
        use_atcr = module_ablation in {"atcr", "full"}

        if use_atcr:
            if ablation == "audio":
                # 只有音频
                s_a = torch.softmax(self.W_a(pooled_audio), dim=-1)
                s_f = s_a
                g_a = torch.ones_like(s_a)
                g_t = torch.zeros_like(s_a)

            elif ablation == "text":
                # 只有文本
                s_t = torch.softmax(self.W_t(global_text_for_factor), dim=-1)
                s_f = s_t
                g_a = torch.zeros_like(s_t)
                g_t = torch.ones_like(s_t)

            else:
                # 音频+文本都有
                s_a = torch.softmax(self.W_a(pooled_audio), dim=-1)
                s_t = torch.softmax(self.W_t(global_text_for_factor), dim=-1)

                gate_input = torch.cat([pooled_audio, global_text_for_factor], dim=-1)
                g_a = torch.sigmoid(self.gate_a_net(gate_input))
                g_t = torch.sigmoid(self.gate_t_net(gate_input))

                s_f = (g_a * s_a + g_t * s_t) / 2.0
        else:
            # 不使用 ATCR：用零向量占位，保持分类头维度不变
            s_f = torch.zeros(B, self.num_factors, device=device, dtype=dtype)
            g_a = None
            g_t = None

        # -------------------------------------------------
        # D. 主干融合与最终分类
        # -------------------------------------------------
        fusion_input_base = torch.cat([pooled_audio, global_text_for_fusion], dim=-1)
        pooled_fused = self.fusion_fc(fusion_input_base)
        features_96 = self.head_hidden(pooled_fused)

        scale_factor = 3.0
        logits_input = torch.cat([features_96, s_f * scale_factor], dim=-1)
        logits = self.head_final(logits_input)

        return {
            "pooled": pooled_fused,
            "logits": logits,
            "alpha": alpha,
            "gate_g": gate_pool_g,
            "sym_factors": s_f,
            "g_a": g_a,
            "g_t": g_t,
        }
