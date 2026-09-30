"""Evaluasi kalibrasi probabilitas model dengan cross-validation."""
import json, numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from clean_data import load_mode, FEATURE_SETS, TARGET
from model import NaiveBayesPCOS

res = {}
for mode, fs in FEATURE_SETS.items():
    df, _ = load_mode(mode); y = df[TARGET].to_numpy(); P, Y = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(df, y):
        P.append(NaiveBayesPCOS(fs["cont"], fs["bin"]).fit(df.iloc[tr]).predict_proba(df.iloc[te])); Y.append(y[te])
    P, Y = np.concatenate(P), np.concatenate(Y)
    bins = [0, .1, .3, .5, .7, .9, 1.0001]; rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (P >= a) & (P < b)
        rel.append(dict(bin=f"{a:.1f}-{min(b,1):.1f}", n=int(m.sum()), mean_pred=float(P[m].mean()) if m.any() else None,
                        frac_pos=float(Y[m].mean()) if m.any() else None))
    res[mode] = dict(brier=float(brier_score_loss(Y, P)), brier_base=float(brier_score_loss(Y, np.full_like(P, Y.mean()))),
                     extreme=float(((P < .05) | (P > .95)).mean()), reliability=rel)
    print(mode, {k: round(v, 3) for k, v in res[mode].items() if k != "reliability"})
    for r in rel: print("  ", r)
json.dump(res, open("outputs/calibration.json", "w"), indent=1)
