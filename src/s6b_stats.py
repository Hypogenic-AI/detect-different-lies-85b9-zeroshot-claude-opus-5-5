"""Stage 6b: paired significance tests and bootstrap CIs for the behavioural results."""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from common import RES, CONDITIONS, jdump

df = pd.read_csv(os.path.join(RES, "labels.csv"))
rng = np.random.default_rng(0)
out = {}
K = df[df.know == "K"]
# paired (same known questions) McNemar exact test: P(false | K) under condition vs placebo
for c in ["sycophancy", "incentive", "instructed"]:
    a = (K[f"g_{c}"] == "incorrect").values
    b = (K["g_placebo"] == "incorrect").values
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    out[f"mcnemar_K_false_{c}_vs_placebo"] = dict(rate_cond=float(a.mean()), rate_placebo=float(b.mean()),
                                                   excess_pp=float(100 * (a.mean() - b.mean())),
                                                   n_cond_only=n10, n_placebo_only=n01, p=float(p))
# bootstrap CI (over questions) for the share of false answers coming from known questions
for c in CONDITIONS:
    f = (df[f"g_{c}"] == "incorrect").values
    k = (df.know == "K").values
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(df), len(df))
        bs.append((f[idx] & k[idx]).sum() / max(1, f[idx].sum()))
    out[f"lie_share_{c}"] = dict(share=float((f & k).sum() / f.sum()), lo=float(np.percentile(bs, 2.5)),
                                 hi=float(np.percentile(bs, 97.5)))
    # strict lie: known, false, AND the honest follow-up recovers the correct answer in-context
    if c != "neutral":
        strict = f & k & (df[f"fu_{c}"] == "correct").values
        out[f"strict_lie_share_{c}"] = dict(n=int(strict.sum()), share=float(strict.sum() / f.sum()))
jdump(out, os.path.join(RES, "behaviour_stats.json"))
print(json.dumps(out, indent=1))
