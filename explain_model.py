import torch
from torch.utils.data import DataLoader
import matplotlib.collections as mcoll
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",

    # 关键：全局字号 8pt
    "font.size": 8,

    # 建议同时明确各元素字号，避免被 seaborn/font_scale 覆盖
    "axes.labelsize": 9,      # 或 9（常见：轴标题略大于刻度）
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})
mpl.rcParams["svg.fonttype"] = "none"


# 引入你的项目代码
from dataset_gate import GateDataset, collate_fn
from model import GateAggWithBiGRUClassifier

# ================= 配置 =================
# [请修改] 你的模型路径 (建议选效果最好的那个)
CKPT_PATH = "best_model/best_model_en1.pt"
# [请修改] 你的测试集列表路径
TEST_TXT = "dataset/list_ADReSSo_fusion/test.txt"
# [请修改] 模型参数 (需与训练时一致)
ARGS = {
    "rnn_hidden": 256,
    "hidden_gate": 256,
    "hidden_main": 96,
    "num_factors": 8,
    "num_classes": 2
}


# =======================================

def extract_interpretability_data(model, loader, device):
    model.eval()

    data_records = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio_embeds"].to(device)
            local = batch["local_text_embeds"].to(device)
            global_t = batch["global_text_embeds"].to(device)
            mask = batch["window_mask"].to(device)
            y = batch["y_main"].to(device)

            # 模型前向传播
            out = model(audio, local, global_t, mask)

            # 提取关键变量
            # g_a, g_t: (B, K) -> 我们取平均值代表该样本对 音频/文本 的整体依赖程度
            # 或者如果是 (B, 1) 就直接取
            g_a = out["g_a"].mean(dim=-1).cpu().numpy()  # (B,)
            g_t = out["g_t"].mean(dim=-1).cpu().numpy()  # (B,)

            # sym_factors: (B, K)
            sym_factors = out["sym_factors"].cpu().numpy()

            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = y.cpu().numpy()

            for i in range(len(labels)):
                data_records.append({
                    "label": labels[i],
                    "pred": preds[i],
                    "g_a": g_a[i],
                    "g_t": g_t[i],
                    "sym_factors": sym_factors[i]  # 这是一个向量
                })

    return data_records



def plot_raincloud_final(df):
    # ================= 0. 数据格式转换 (Wide -> Long) =================
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({"label": row['label'], "Gate Value": row['g_a'], "Modality": "Audio Gate"})
        plot_data.append({"label": row['label'], "Gate Value": row['g_t'], "Modality": "Text Gate"})
    df_long = pd.DataFrame(plot_data)

    # ================= 1. 标签映射与重排序 =================
    label_map = {0: "AD", 1: "HC"}
    df_long['Label_Text'] = df_long['label'].map(label_map)
    order_list = ["HC", "AD"]

    # ================= 2. 绘图设置 =================
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    my_pal = {"Audio Gate": "#0072B2", "Text Gate": "#D55E00"}

    # ================= 3. 绘制 RainCloud：互换坐标轴 =================
    pt.RainCloud(
        x="Label_Text",  # x 放数值
        y="Gate Value",  # y 放类别Label_Text
        hue="Modality",
        data=df_long,
        order=order_list,
        palette=my_pal,
        bw=.2,
        jitter=0.20,
        width_viol=.8,
        ax=ax,
        orient="h",  # 关键：水平显示 => 类别在 y，数值在 x（实现坐标轴互换）
        alpha=0.65,
        dodge=True,
        pointplot=False,
        move=.2,
        box_showmeans=False,
        linecolor='black',
        point_size=1.8
    )

    # 统一所有 Line2D（须线/中位线等）
    for ln in ax.findobj(mpl.lines.Line2D):
        if ln.get_linestyle() != "None":
            ln.set_color("black")
            ln.set_linewidth(0.75)

    # 统一 LineCollection（有些须线/边框可能在这里）
    for coll in ax.findobj(mcoll.LineCollection):
        coll.set_color("black")
        coll.set_linewidth(0.75)

    for coll in ax.findobj(mcoll.PolyCollection):
        coll.set_edgecolor("black")
        coll.set_linewidth(0.75)

    # 统一 Patch 的边缘（箱体、violin 轮廓有时是 patch）
    for p in ax.findobj(mpl.patches.Patch):
        # 避免误伤图例：可以跳过 legend 的 patch（可选）
        if p.get_alpha() is not None:  # 仅处理绘图区内的实体
            p.set_edgecolor("black")
            p.set_linewidth(0.75)

    # ================= 3.1 去除“小圆圈”（离群点） =================
    for ln in ax.lines:
        if ln.get_marker() == 'o' and ln.get_linestyle() == 'None':
            ln.set_markersize(0)
            ln.set_alpha(0)

    '''# ================= 3.2 控制雨点散点“可读性” =================
    for coll in ax.collections:
        # 雨点散点通常是 PathCollection，且点数较多
        if hasattr(coll, "get_offsets") and len(coll.get_offsets()) > 20:
            coll.set_alpha(0.75)  # 0.5~0.8 自行微调
            coll.set_sizes([8])  # 点大小（6~10 常用）
            # coll.set_rasterized(True)  # 可选：PDF/SVG 文件不至于过大
    '''
    sns.despine()

    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(0.75)
    ax.tick_params(axis="both", colors="black", width=0.75, length=4, labelsize=8)

    # 轴标签互换后：x 是 Gate，y 是类别
    ax.set_xlabel("Gate Activation Strength", fontsize=8, labelpad=6)
    ax.set_ylabel("Diagnostic Group", fontsize=8, labelpad=6)

    plt.tight_layout()

    # ---- tight_layout 之后再放 legend：右上角空白处，不占画布 ----
    handles = [
        Patch(facecolor=my_pal["Audio Gate"], edgecolor="none", label="Audio Gate"),
        Patch(facecolor=my_pal["Text Gate"], edgecolor="none", label="Text Gate"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.02, 1.00),
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        handlelength=1.4,
        handleheight=0.7,
        labelspacing=0.25,
        handletextpad=0.5
    )

    save_name = "fig3-1"
    plt.savefig(save_name, dpi=900, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(save_name.replace(".png", ".svg"),
                format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"最终整洁版云雨图已生成：{save_name}")






def plot_symptom_fingerprint(df, num_factors=8):
    """图2：症状因子指纹热力图"""
    # 计算每个类别在每个因子上的平均激活值
    num_classes = 2
    heatmap_data = np.zeros((num_classes, num_factors))

    for cls in range(num_classes):
        cls_data = df[df['label'] == cls]
        if len(cls_data) > 0:
            # 堆叠所有的 factors 向量
            factors_stack = np.stack(cls_data['sym_factors'].values)
            # 计算平均
            heatmap_data[cls] = factors_stack.mean(axis=0)

    plt.figure(figsize=(10, 1.33))  # 高度稍微调小一点，因为只有2行

    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f",
                yticklabels=["Class 0 (AD)", "Class 1 (HC)"],
                xticklabels=[f"F{i + 1}" for i in range(num_factors)])


    plt.xlabel("Latent Factors (Learnable Symptoms)")
    plt.ylabel("Ground Truth")

    save_path = "fig4-1"
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"图2已保存: {save_path}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading data...")
    test_set = GateDataset(TEST_TXT)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 2. 加载模型
    print(f"Loading model from {CKPT_PATH}...")
    model = GateAggWithBiGRUClassifier(
        audio_dim=768, text_dim=768,
        rnn_hidden=ARGS["rnn_hidden"],
        hidden_gate=ARGS["hidden_gate"],
        hidden_main=ARGS["hidden_main"],
        num_main_classes=ARGS["num_classes"],
        num_factors=ARGS["num_factors"]
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)

    # 3. 提取数据
    print("Running inference for interpretability...")
    raw_data = extract_interpretability_data(model, test_loader, device)
    df = pd.DataFrame(raw_data)

    print(f"Extracted {len(df)} samples.")

    # 4. 绘图
    # 设置绘图风格
    sns.set_theme(style="white")

    # 绘制图1：门控分布
    plot_raincloud_final(df)  # 推荐先用 False


    # 绘制图2：因子热力图
    plot_symptom_fingerprint(df, num_factors=ARGS["num_factors"])

    print("\n可视化分析完成！请查看生成的 .png 图片。")


if __name__ == "__main__":
    main()