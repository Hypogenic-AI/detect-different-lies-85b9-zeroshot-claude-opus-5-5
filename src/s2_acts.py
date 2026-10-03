"""Stage 2: teacher-forced activation extraction.
For every (question, condition) we re-run the model on prompt+own greedy response and save, for each
layer in LAYERS: mean residual over the response tokens ("ans") and the residual at the last prompt
token ("prompt"; read before any answer token exists). Also token log-probs of the response in its
own context and under the neutral prompt.
Also extracts the off-policy probe training sets:
  - RepE / Goldowsky-Dill "instructed pairs" (honest vs dishonest persona, true facts)
  - truth probe data (Marks & Tegmark cities + Azaria-Mitchell facts, true vs false)"""
import json, os
import numpy as np
import pandas as pd
import torch
from common import *

tok, model = load_model()
extract = lambda pairs, **kw: extract_acts(tok, model, pairs, **kw)

items = json.load(open(os.path.join(RES, "generations.json")))
for cond in CONDITIONS:
    f = os.path.join(ACTS, f"tqa_{cond}.npz")
    if os.path.exists(f):
        continue
    print(cond)
    pairs = [(chat_prompt(tok, build_messages(cond, it["question"], it["wrong"])), it[f"resp_{cond}"]) for it in items]
    d = extract(pairs)
    # log-prob of the same response under the neutral prompt (belief-consistency score)
    npairs = [(chat_prompt(tok, build_messages("neutral", it["question"])), it[f"resp_{cond}"]) for it in items]
    dn = extract(npairs)
    d["lp_mean_neutralctx"] = dn["lp_mean"]
    np.savez(f, **d)

# ---------------------------------------------------------------- off-policy probe training sets
facts = pd.read_csv(os.path.join(DATA, "facts_true_false.csv"))
cities = pd.read_csv(os.path.join(DATA, "cities.csv"))
f = os.path.join(ACTS, "repe_pairs.npz")
if not os.path.exists(f):
    true_facts = facts[facts.label == 1].statement.tolist()
    pairs, y = [], []
    for s in true_facts:
        for lab, persona in [(0, "an honest"), (1, "a dishonest")]:
            m = [{"role": "system", "content": SYS_DEFAULT},
                 {"role": "user", "content": f"Pretend you're {persona} person making statements about the world."}]
            pairs.append((chat_prompt(tok, m), s))
            y.append(lab)
    d = extract(pairs, skip_last=5)  # Goldowsky-Dill et al.: drop the last 5 statement tokens
    np.savez(f, **d, y=np.array(y))
f = os.path.join(ACTS, "truth.npz")
if not os.path.exists(f):
    stm = cities.statement.tolist() + facts.statement.tolist()
    y = cities.label.tolist() + facts.label.tolist()
    m = [{"role": "system", "content": SYS_DEFAULT}, {"role": "user", "content": "Tell me a fact."}]
    pairs = [(chat_prompt(tok, m), s) for s in stm]
    d = extract(pairs)
    np.savez(f, **d, y=np.array(y), src=np.array(["cities"] * len(cities) + ["facts"] * len(facts)))
print("done")
