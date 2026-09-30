import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import NaiveBayesPCOS
from sklearn.naive_bayes import GaussianNB, BernoulliNB


def _toy(n=300, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    return pd.DataFrame({"a": r.normal(y * 2, 1), "b": r.normal(-y, 2),
                         "s1": (r.random(n) < 0.2 + 0.5 * y).astype(int),
                         "s2": (r.random(n) < 0.6 - 0.4 * y).astype(int), "y": y})


def test_gaussian_matches_sklearn():
    d = _toy()
    ours = NaiveBayesPCOS(["a", "b"], [], target="y").fit(d).predict_proba(d)
    ref = GaussianNB().fit(d[["a", "b"]], d["y"]).predict_proba(d[["a", "b"]])[:, 1]
    assert np.allclose(ours, ref, atol=1e-6)


def test_bernoulli_matches_sklearn():
    d = _toy()
    ours = NaiveBayesPCOS([], ["s1", "s2"], target="y").fit(d).predict_proba(d)
    ref = BernoulliNB(alpha=1.0).fit(d[["s1", "s2"]], d["y"]).predict_proba(d[["s1", "s2"]])[:, 1]
    assert np.allclose(ours, ref, atol=1e-9)


def test_explain_sums_to_log_odds():
    d = _toy()
    m = NaiveBayesPCOS(["a", "b"], ["s1", "s2"], target="y").fit(d)
    p = m.predict_proba(d.iloc[[0]])[0]
    assert np.isclose(m.explain(d.iloc[0])["kontribusi"].sum(), np.log(p / (1 - p)), atol=1e-6)
