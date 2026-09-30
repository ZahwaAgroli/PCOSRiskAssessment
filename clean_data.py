"""Pembersihan data PCOS dan definisi set fitur.

Data berasal dari Kaggle "PCOS (without infertility)",
dengan 541 pasien dari 10 rumah sakit di Kerala, India.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/Data_PCOS_Kaggle.xlsx"
TARGET = "PCOS (Y/N)"

BIN_COMMON = ["Cycle(R/I)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Pimples(Y/N)",
              "Weight gain(Y/N)", "Fast food (Y/N)", "Hair loss(Y/N)"]
FEATURE_SETS = {
    # mode klinis: butuh hasil USG (jumlah folikel) dan AMH
    "klinis": dict(cont=["Follicle No. (L)", "Follicle No. (R)", "AMH(ng/mL)", "BMI",
                         "Cycle length(days)", "Age (yrs)"], bin=BIN_COMMON),
    # mode mandiri: bisa diisi tanpa USG / lab
    "mandiri": dict(cont=["BMI", "Cycle length(days)", "Age (yrs)"], bin=BIN_COMMON),
}


def load_clean(cont, binf, path=RAW_PATH):
    """Kembalikan (df_bersih, log). Hanya kolom yang dipakai yang diperiksa."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    log = [("Baris mentah", len(df))]
    use = cont + binf

    if "AMH(ng/mL)" in use:
        df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors="coerce")
        log.append(("AMH berisi teks non-angka ('a') -> dibuang", int(df["AMH(ng/mL)"].isna().sum())))
        big = df["AMH(ng/mL)"] > 100         
        log.append(("AMH > 100 ng/mL (2333 ng/mL) -> dibuang", int(big.sum())))
        df.loc[big, "AMH(ng/mL)"] = np.nan

    if "FSH/LH" in use:                  
        big = df["FSH/LH"] > 100            
        log.append(("FSH/LH > 100 -> dibuang", int(big.sum())))
        df.loc[big, "FSH/LH"] = np.nan

    if "Cycle(R/I)" in use:
        bad = ~df["Cycle(R/I)"].isin([2, 4])
        log.append(("Cycle(R/I) di luar {2,4} (nilai 5) -> dibuang", int(bad.sum())))
        df["Cycle(R/I)"] = df["Cycle(R/I)"].map({2: 0, 4: 1})   # 0=teratur, 1=tidak teratur

    out = df[use + [TARGET]].apply(pd.to_numeric, errors="coerce")
    miss = out.isna().any(axis=1)
    out = out.dropna().reset_index(drop=True)
    log.append(("Baris dibuang total (missing/invalid)", int(miss.sum())))
    log.append(("Baris akhir", len(out)))
    log.append(("Kelas negatif / positif", f"{int((out[TARGET]==0).sum())} / {int((out[TARGET]==1).sum())}"))
    return out, log


def load_mode(mode):
    fs = FEATURE_SETS[mode]
    return load_clean(fs["cont"], fs["bin"])


if __name__ == "__main__":
    for m in FEATURE_SETS:
        print(f"== mode {m} ==")
        for k, v in load_mode(m)[1]:
            print(f"  {k}: {v}")
