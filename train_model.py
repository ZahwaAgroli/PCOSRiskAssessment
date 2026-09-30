"""
Evaluasi model, ablasi fitur, dan pelatihan model akhir.
Hasil evaluasi disimpan dalam outputs/results.json.
"""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix
from clean_data import FEATURE_SETS, BIN_COMMON, TARGET, load_clean, load_mode
from model import NaiveBayesPCOS

SEED, THRESHOLDS = 42, (0.3, 0.4, 0.5)
ABLATION = {
    "Set awal (14 fitur: +FSH/LH, WHR, olahraga)": dict(
        cont=["BMI", "AMH(ng/mL)", "Waist:Hip Ratio", "Cycle length(days)", "FSH/LH", "Follicle No. (L)", "Follicle No. (R)"],
        bin=["Cycle(R/I)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Pimples(Y/N)", "Weight gain(Y/N)",
             "Fast food (Y/N)", "Reg.Exercise(Y/N)"]),
    "Klinis (13 fitur)": FEATURE_SETS["klinis"],
    "Mandiri (10 fitur, tanpa USG/lab)": FEATURE_SETS["mandiri"],
    "Hanya jumlah folikel (2 fitur)": dict(cont=["Follicle No. (L)", "Follicle No. (R)"], bin=[]),
}


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"), tp=tp, tn=tn, fp=fp, fn=fn)


def cross_validate(df, cont, binf, kind="hybrid"):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    y = df[TARGET].to_numpy()
    rows, thr_rows, cm0, example = [], [], np.zeros((2, 2), int), None
    for i, (tr, te) in enumerate(rskf.split(df, y)):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        if kind == "hybrid":
            m = NaiveBayesPCOS(cont, binf).fit(d_tr); p = m.predict_proba(d_te)
        elif kind == "sklearn":
            X = cont + binf; p = GaussianNB().fit(d_tr[X], y[tr]).predict_proba(d_te[X])[:, 1]
        else:  # baseline mayoritas
            p = np.full(len(te), float(y[tr].mean() >= 0.5))
        r = metrics(y[te], p); rows.append(r)
        thr_rows.append({t: metrics(y[te], p, t) for t in THRESHOLDS})
        if i < 5:
            cm0 += np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])     # repeat ke-0 = 5 fold pertama
        if kind == "hybrid" and example is None:
            hit = np.where((y[te] == 1) & (p > 0.75) & (p < 0.97))[0]
            if len(hit):
                row = d_te.iloc[hit[0]]
                example = dict(prob=float(p[hit[0]]), true=int(y[te][hit[0]]),
                               contrib=m.explain(row).to_dict("records"))
    R = pd.DataFrame(rows)
    out = {k: [float(R[k].mean()), float(R[k].std())] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
    out["cm"] = cm0.tolist()
    out["thresholds"] = {str(t): {k: float(np.mean([x[t][k] for x in thr_rows])) for k in ["precision", "recall", "f1"]}
                         for t in THRESHOLDS}
    return out, example


if __name__ == "__main__":
    res = {"cleaning": {}, "cv": {}, "ablation": {}, "example": None}
    for mode in FEATURE_SETS:
        df, log = load_mode(mode); fs = FEATURE_SETS[mode]
        res["cleaning"][mode] = [[a, b] for a, b in log]
        res["cv"][mode] = {}
        for kind in ("baseline", "sklearn", "hybrid"):
            r, ex = cross_validate(df, fs["cont"], fs["bin"], kind); res["cv"][mode][kind] = r
            if kind == "hybrid" and mode == "klinis":
                res["example"] = ex
        # model final (seluruh data) untuk aplikasi
        final = NaiveBayesPCOS(fs["cont"], fs["bin"]).fit(df)
        pickle.dump(final, open(f"model/pcos_{mode}.pkl", "wb"))
        res.setdefault("final_params", {})[mode] = dict(
            prior_pos=float(final.prior[1]), n=len(df))
    for name, fs in ABLATION.items():
        df, _ = load_clean(fs["cont"], fs["bin"])
        r, _ = cross_validate(df, fs["cont"], fs["bin"], "hybrid")
        res["ablation"][name] = dict(n=len(df), **{k: r[k] for k in ["accuracy", "recall", "f1", "auc"]})
    def clean(o):  # NaN bukan JSON valid -> None
        if isinstance(o, dict): return {a: clean(b) for a, b in o.items()}
        if isinstance(o, list): return [clean(b) for b in o]
        return None if isinstance(o, float) and o != o else o
    json.dump(clean(res), open("outputs/results.json", "w"), indent=1, ensure_ascii=False)

    # GRAFIK HASIL EVALUASI
    C = {"neg": "#2A9D8F", "pos": "#B5385C", "ink": "#22303C", "grey": "#9AA5B1"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, mode in zip(axs, FEATURE_SETS):
        cm = np.array(res["cv"][mode]["hybrid"]["cm"])
        ax.imshow(cm, cmap="Purples")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else C["ink"])
        ax.set_xticks([0, 1], ["Prediksi -", "Prediksi +"]); ax.set_yticks([0, 1], ["Aktual -", "Aktual +"])
        ax.set_title(f"Mode {mode}"); ax.spines[:].set_visible(False)
    plt.tight_layout(); plt.savefig("outputs/fig_confusion.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(8, 3.4)); w = 0.2; mets = ["accuracy", "recall", "f1", "auc"]
    labels = ["Baseline", "GaussianNB (sklearn)", "Hybrid NB - mandiri", "Hybrid NB - klinis"]
    sel = [("klinis", "baseline"), ("klinis", "sklearn"), ("mandiri", "hybrid"), ("klinis", "hybrid")]
    cols = [C["grey"], "#E9C46A", "#E76F51", C["pos"]]
    for k, ((mode, kind), lab, col) in enumerate(zip(sel, labels, cols)):
        vals = [res["cv"][mode][kind][m][0] * 100 for m in mets]
        ax.bar(np.arange(4) + (k - 1.5) * w, vals, w, label=lab, color=col)
    ax.set_xticks(range(4), ["Akurasi", "Recall", "F1", "AUC"]); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center"); plt.tight_layout()
    plt.savefig("outputs/fig_compare.png", dpi=200); plt.close()

    ex = pd.DataFrame(res["example"]["contrib"]).head(9).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(ex["fitur"], ex["kontribusi"], color=[C["pos"] if v > 0 else C["neg"] for v in ex["kontribusi"]])
    ax.axvline(0, color=C["ink"], lw=0.8); ax.set_xlabel("Kontribusi log-likelihood ratio (nats)")
    ax.set_title(f"Contoh pasien uji: P(PCOS) = {res['example']['prob']:.2f}"); plt.tight_layout()
    plt.savefig("outputs/fig_example.png", dpi=200); plt.close()
    print(json.dumps({m: {k: [round(x*100, 1) for x in v["accuracy"::0]] if False else
                          {kk: round(v[kk][0]*100, 1) for kk in ["accuracy", "precision", "recall", "f1", "auc"]}
                          for k, v in res["cv"][m].items()} for m in res["cv"]}, indent=1))
