"""Stage 7b: MASK deception-probe AUROC (lie vs honest-correct) within each MASK subset, to check whether
the pooled AUROC reflects differences between prompt archetypes rather than lying."""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import RES, ACTS, LAYERS, jdump

LI = LAYERS.index(16)
df = pd.read_csv(os.path.join(RES, "mask_labels.csv"))
A = np.load(os.path.join(ACTS, "mask_pressure.npz"))
probes = pickle.load(open(os.path.join(ACTS, "probes_L16.pkl"), "rb"))
out = []
for name in ["deception probe (RepE pairs)", "truth probe (cities+facts)", "lie probe (within-prompt)"]:
    s = probes[name].decision_function(A["ans"][:, LI].astype(np.float32))
    for sub in ["pooled"] + sorted(df.subset.unique()):
        m = np.ones(len(df), bool) if sub == "pooled" else (df.subset == sub).values
        pos, neg = s[m & (df.cat == "lie").values], s[m & (df.cat == "honest correct").values]
        out.append(dict(detector=name, subset=sub, n_lie=len(pos), n_honest=len(neg),
                        auc=float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
                        if len(pos) >= 5 and len(neg) >= 5 else None))
pd.DataFrame(out).to_csv(os.path.join(RES, "mask_auroc_by_subset.csv"), index=False)
print(pd.DataFrame(out).to_string())
