"""Visualisasi EDA untuk mode klinis & mandiri"""
import math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from clean_data import load_mode, FEATURE_SETS, TARGET

C = {0: "#2A9D8F", 1: "#B5385C"}
sns.set_style("whitegrid"); plt.rcParams.update({"font.size": 9})

for mode in FEATURE_SETS:                      
    df, _ = load_mode(mode)
    fs = FEATURE_SETS[mode]
    n = len(fs["cont"])
    ncol = 3
    nrow = math.ceil(n / ncol)

    # 1) sebaran fitur kontinu per kelas
    fig, axs = plt.subplots(nrow, ncol, figsize=(9, 2.4 * nrow), squeeze=False)
    for i, col in enumerate(fs["cont"]):
        ax = axs.flat[i]
        for c in (0, 1):
            sns.kdeplot(df[df[TARGET] == c][col], ax=ax, fill=True, color=C[c], alpha=.4,
                        label="PCOS" if c else "Non-PCOS", warn_singular=False)
        ax.set_title(col, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    for j in range(n, nrow * ncol):              # kosongkan slot yang tidak terpakai
        axs.flat[j].axis("off")
    axs.flat[0].legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"outputs/eda_{mode}_continuous.png", dpi=200); plt.close()

    # 2) prevalensi fitur biner per kelas
    fig, ax = plt.subplots(figsize=(6, 3.4))
    prev = df.groupby(TARGET)[fs["bin"]].mean().T * 100
    prev.columns = ["Non-PCOS", "PCOS"]
    prev.sort_values("PCOS").plot.barh(ax=ax, color=[C[0], C[1]], width=.75)
    ax.set_xlabel("% pasien dengan gejala (Cycle(R/I): tidak teratur)")
    ax.set_title(f"Mode {mode}"); plt.tight_layout()
    plt.savefig(f"outputs/eda_{mode}_binary.png", dpi=200); plt.close()

    # 3) korelasi antar fitur
    fig, ax = plt.subplots(figsize=(6.4, 5))
    corr = df[fs["cont"] + fs["bin"] + [TARGET]].corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax, cbar=False)
    ax.set_title(f"Mode {mode}"); plt.tight_layout()
    plt.savefig(f"outputs/eda_{mode}_corr.png", dpi=200); plt.close()

    print(f"=== Mode {mode} ===")
    print(corr[TARGET].drop(TARGET).round(2).sort_values(key=abs, ascending=False).to_dict())
    print()