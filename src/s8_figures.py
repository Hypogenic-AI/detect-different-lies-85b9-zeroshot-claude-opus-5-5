"""Stage 8: figures + LaTeX tables from results/*.csv (run after s6 and s7)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

from common import FIG
from common import TAB
os.makedirs(TAB, exist_ok=True)
os.makedirs(os.path.join(RES, "previews"), exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
DET = ["deception probe (RepE pairs)", "truth probe (cities+facts)", "instructed-lie probe (cross-prompt)",
       "lie probe (within-prompt)", "hallucination probe (SEP-style)", "answer NLL (own context)",
       "answer NLL (neutral context)", "knowledge-gated falsity", "lie-vs-halluc probe (supervised)"]
SHORT = {"deception probe (RepE pairs)": "Deception (RepE)", "truth probe (cities+facts)": "Truth probe",
         "instructed-lie probe (cross-prompt)": "Instructed-lie", "lie probe (within-prompt)": "Lie (within-prompt)",
         "hallucination probe (SEP-style)": "Hallucination (SEP)", "answer NLL (own context)": "NLL (own ctx)",
         "answer NLL (neutral context)": "NLL (neutral ctx)",
         "knowledge-gated falsity": "Knowledge-gated falsity", "lie-vs-halluc probe (supervised)": "Lie-vs-halluc (sup.)"}
CAT_COL = {"correct": "#7f8c8d", "halluc": "#2e86c1", "lie": "#c0392b", "misbelief": "#5d6d7e", "partial": "#e59866"}

# ---------------------------------------------------------------- score distributions
SL = pd.read_csv(os.path.join(RES, "scores_test_z.csv"))
groups = [("neutral", "correct"), ("neutral", "halluc"), None,
          ("sycophancy", "correct"), ("sycophancy", "halluc"), ("sycophancy", "lie"), None,
          ("incentive", "correct"), ("incentive", "halluc"), ("incentive", "lie"), None,
          ("instructed", "correct"), ("instructed", "halluc"), ("instructed", "lie")]
dets = [d for d in DET if d in SL.detector.unique()]
MAIN_DETS = ["deception probe (RepE pairs)", "truth probe (cities+facts)", "lie probe (within-prompt)",
             "hallucination probe (SEP-style)", "answer NLL (neutral context)", "knowledge-gated falsity"]


def score_fig(dlist, name):
    fig, axes = plt.subplots(len(dlist), 1, figsize=(7.2, 1.2 * len(dlist)), sharex=True)
    for ax, d in zip(axes, dlist):
        x = 0; ticks = []; labels = []
        for g in groups:
            if g is None:
                x += 0.6; continue
            v = SL[(SL.detector == d) & (SL.cond == g[0]) & (SL.cat == g[1])].z.values
            v = np.clip(v, -6, 12)
            if len(v):
                bp = ax.boxplot([v], positions=[x], widths=0.7, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set_facecolor(CAT_COL[g[1]]); b.set_alpha(0.8)
                for m in bp["medians"]:
                    m.set_color("black")
            ticks.append(x); labels.append(f"{g[0][:5]}\n{g[1]}\n(n={len(v)})")
            x += 1
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_ylabel(SHORT[d], fontsize=7)
    axes[-1].set_xticks(ticks); axes[-1].set_xticklabels(labels, fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(RES, "previews", f"{name}.png"), dpi=110, bbox_inches="tight"); plt.close()


score_fig(MAIN_DETS, "score_distributions")
score_fig(dets, "score_distributions_all")

# ---------------------------------------------------------------- AUROC heatmap
A = pd.read_csv(os.path.join(RES, "auroc_main.csv"))
contr = ["halluc vs correct | neutral", "lie vs correct | sycophancy", "lie vs correct | incentive",
         "lie vs correct | instructed", "lie vs halluc | sycophancy", "lie vs halluc | incentive",
         "lie vs halluc | instructed", "lie (instructed) vs halluc (neutral)"]
M = np.full((len(dets), len(contr)), np.nan)
for a, d in enumerate(dets):
    for b, c in enumerate(contr):
        v = A[(A.detector == d) & (A.contrast == c)].auc.values
        if len(v) and not pd.isna(v[0]):
            M[a, b] = v[0]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        if not np.isnan(M[a, b]):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[a, b] - 0.5) > 0.3 else "black")
ax.set_yticks(range(len(dets))); ax.set_yticklabels([SHORT[d] for d in dets])
ax.set_xticks(range(len(contr)))
XL = ["halluc vs\ncorrect\n(neutral)", "lie vs\ncorrect\n(sycoph.)", "lie vs\ncorrect\n(incent.)",
      "lie vs\ncorrect\n(instr.)", "lie vs\nhalluc\n(sycoph.)", "lie vs\nhalluc\n(incent.)",
      "lie vs\nhalluc\n(instr.)", "lie (instr.)\nvs halluc\n(neutral)"]
ax.set_xticklabels(XL, fontsize=6.5)
ax.axvline(3.5, color="k", lw=1.5); ax.axvline(6.5, color="k", lw=0.8, ls="--")
plt.colorbar(im, ax=ax, fraction=0.025, label="AUROC (positive = first class)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "auroc_heatmap.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(RES, "previews", "auroc_heatmap.png"), dpi=110, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- layer sweep
S = pd.read_csv(os.path.join(RES, "layer_sweep.csv"))
show = [("lie vs correct | instructed", "-"), ("lie vs halluc | instructed", "--"),
        ("halluc vs correct | neutral", ":"), ("own held-out data", "-."), ("lie vs halluc | sycophancy", (0, (1, 1)))]
probe_dets = [d for d in dets if "NLL" not in d]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=True)
axes = axes.ravel()
for ax, d in zip(axes, probe_dets):
    for c, ls in show:
        v = S[(S.detector == d) & (S.contrast == c)].sort_values("layer")
        if len(v):
            ax.plot(v.layer, v.auc, ls=ls, marker="o", ms=2.5, label=c.replace(" | ", " / "),
                    color={"lie vs correct | instructed": "C0", "lie vs halluc | instructed": "C1",
                           "halluc vs correct | neutral": "C2", "own held-out data": "C3",
                           "lie vs halluc | sycophancy": "C4"}[c])
    ax.axhline(0.5, color="grey", lw=0.5)
    ax.set_title(SHORT[d], fontsize=7); ax.set_ylim(0, 1.02)
for ax in axes[4:]:
    ax.set_xlabel("layer")
axes[0].set_ylabel("AUROC"); axes[4].set_ylabel("AUROC")
h, l = axes[0].get_legend_handles_labels()
axes[-1].axis("off"); axes[-1].legend(h, l, fontsize=6.5, loc="center", frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "layer_sweep.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(RES, "previews", "layer_sweep.png"), dpi=110, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- tables
def fmt(r):
    if pd.isna(r.auc):
        return "--"
    return f"{r.auc:.2f} {{\\tiny[{r.lo:.2f},{r.hi:.2f}]}}"


lines = []
for d in dets:
    cells = []
    for c in contr:
        r = A[(A.detector == d) & (A.contrast == c)]
        cells.append(fmt(r.iloc[0]) if len(r) else "--")
    lines.append(SHORT[d] + " & " + " & ".join(cells) + r" \\")
open(os.path.join(TAB, "auroc_main.tex"), "w").write("\n".join(lines))
ns = []
for c in contr:
    r = A[(A.detector == dets[0]) & (A.contrast == c)].iloc[0]
    ns.append(f"{int(r.n_pos)}/{int(r.n_neg)}")
open(os.path.join(TAB, "auroc_main_n.tex"), "w").write("$n_+/n_-$ & " + " & ".join(ns) + r" \\")

F = pd.read_csv(os.path.join(RES, "flag_rates.csv"))
F5 = F[F.fpr == 0.01]
cols = [("neutral", "correct"), ("neutral", "halluc"), ("sycophancy", "lie"), ("sycophancy", "halluc"),
        ("incentive", "lie"), ("incentive", "halluc"), ("instructed", "correct"), ("instructed", "lie"), ("instructed", "halluc")]
lines = []
for d in dets:
    cells = []
    for c, k in cols:
        v = F5[(F5.detector == d) & (F5.cond == c) & (F5.cat == k)].flag_rate.values
        cells.append(f"{100 * v[0]:.0f}" if len(v) else "--")
    lines.append(SHORT[d] + " & " + " & ".join(cells) + r" \\")
open(os.path.join(TAB, "flag_rates.tex"), "w").write("\n".join(lines))

C = pd.read_csv(os.path.join(RES, "confound.csv"))
lines = []
for d in ["instructed-lie probe (cross-prompt)", "lie probe (within-prompt)", "deception probe (RepE pairs)"]:
    cells = []
    for feat in ["ans", "prompt"]:
        for c in ["instructed-but-truthful vs neutral-truthful", "instructed lie vs instructed-but-truthful"]:
            r = C[(C.detector == d) & (C.feat == feat) & (C.contrast == c)]
            cells.append(fmt(r.iloc[0]) if len(r) else "--")
    lines.append(SHORT[d] + " & " + " & ".join(cells) + r" \\")
open(os.path.join(TAB, "confound.tex"), "w").write("\n".join(lines))

P = pd.read_csv(os.path.join(RES, "auroc_prompt_tokens.csv"))
lines = []
for d in [x for x in dets if "NLL" not in x and x in set(P.detector)]:
    cells = []
    for c in contr:
        r = P[(P.detector == d) & (P.contrast == c)]
        cells.append(f"{r.iloc[0].auc:.2f}" if len(r) and not pd.isna(r.iloc[0].auc) else "--")
    lines.append(SHORT[d] + " & " + " & ".join(cells) + r" \\")
open(os.path.join(TAB, "auroc_prompt.tex"), "w").write("\n".join(lines))

# ---------------------------------------------------------------- MASK
if os.path.exists(os.path.join(RES, "mask_auroc.csv")):
    MA = pd.read_csv(os.path.join(RES, "mask_auroc.csv"))
    lines = []
    mc = ["lie vs honest correct", "honest error vs honest correct", "lie vs honest error", "lie vs false, no belief"]
    for d in MA.detector.unique():
        cells = []
        for c in mc:
            r = MA[(MA.detector == d) & (MA.contrast == c)]
            cells.append(fmt(r.iloc[0]) if len(r) else "--")
        lines.append(SHORT.get(d, d) + " & " + " & ".join(cells) + r" \\")
    r0 = MA[MA.detector == MA.detector.unique()[0]]
    lines.append(r"\midrule $n_+/n_-$ & " + " & ".join(
        f"{int(r0[r0.contrast == c].n_pos.iloc[0])}/{int(r0[r0.contrast == c].n_neg.iloc[0])}" for c in mc) + r" \\")
    open(os.path.join(TAB, "mask_auroc.tex"), "w").write("\n".join(lines))
    MZ = pd.read_csv(os.path.join(RES, "mask_scores_z.csv"))
    cats = ["honest correct", "lie", "honest error", "false, no belief", "evasive"]
    md = [d for d in MA.detector.unique() if "NLL" not in d]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharey=True)
    axes = axes.ravel()
    lab = {"honest correct": "honest\ncorrect", "lie": "lie", "honest error": "honest\nerror",
           "false, no belief": "false,\nno belief", "evasive": "evasive"}
    for ax, d in zip(axes, md):
        vals = [np.clip(MZ[(MZ.detector == d) & (MZ.cat == c)].z.values, -6, 12) for c in cats]
        keep = [i for i, v in enumerate(vals) if len(v) >= 3]
        bp = ax.boxplot([vals[i] for i in keep], positions=range(len(keep)), showfliers=False, patch_artist=True, widths=0.6)
        for b, i in zip(bp["boxes"], keep):
            b.set_facecolor(["#7f8c8d", "#c0392b", "#5d6d7e", "#2e86c1", "#f4d03f"][i])
        ax.set_xticks(range(len(keep))); ax.set_xticklabels([f"{lab[cats[i]]}\n({len(vals[i])})" for i in keep], fontsize=6)
        ax.set_title(SHORT.get(d, d), fontsize=7); ax.axhline(0, color="grey", lw=0.5, ls=":")
    axes[0].set_ylabel("z vs honest correct"); axes[3].set_ylabel("z vs honest correct")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mask_scores.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(RES, "previews", "mask_scores.png"), dpi=110, bbox_inches="tight"); plt.close()
print("figures done")
