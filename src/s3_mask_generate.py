"""Stage 3: MASK (cais/MASK) with the subject model: greedy pressure responses + belief elicitation
(belief_elicit_1 x3 [1 greedy + 2 sampled], belief_elicit_2, belief_elicit_3 greedy, neutral system prompt).
Output: results/mask_generations.json"""
import os
import pandas as pd
from common import *

SUBSETS = ["known_facts", "doubling_down_known_facts", "disinformation", "continuations"]
rows = []
for s in SUBSETS:
    m = pd.read_parquet(os.path.join(DATA, f"mask_{s}.parquet"))
    for _, r in m.iterrows():
        msgs = [{"role": "system", "content": r["system_prompt"]}]
        if s.startswith("doubling_down"):
            msgs += [{"role": "user", "content": r["dd_turn_1_prompt"]},
                     {"role": "assistant", "content": r["dd_turn_1_response"]}]
        msgs.append({"role": "user", "content": r["user_prompt"]})
        rows.append(dict(task_id=r["task_id"], subset=s, proposition=r["proposition"],
                         ground_truth=r["ground_truth"], formatted_ground_truth=r["formatted_ground_truth"],
                         pressure_messages=msgs,
                         belief_prompts=[r[f"belief_elicit_{k}"] for k in (1, 2, 3)]))

tok, model = load_model()
BSYS = "You are a helpful assistant."
pp = [chat_prompt(tok, r["pressure_messages"]) for r in rows]
print("pressure")
for r, o in zip(rows, generate(tok, model, pp, max_new_tokens=320, bs=48)):
    r["pressure_response"] = o
bp = lambda q: chat_prompt(tok, [{"role": "system", "content": BSYS}, {"role": "user", "content": q}])
print("belief greedy")
for k in range(3):
    outs = generate(tok, model, [bp(r["belief_prompts"][k]) for r in rows], max_new_tokens=320, bs=48)
    for r, o in zip(rows, outs):
        r.setdefault("belief_responses", {})[f"b{k + 1}_greedy"] = o
print("belief samples")
outs = generate(tok, model, [bp(r["belief_prompts"][0]) for r in rows], max_new_tokens=320, do_sample=True, n=2, bs=24)
for r, o in zip(rows, outs):
    r["belief_responses"]["b1_s1"], r["belief_responses"]["b1_s2"] = o
jdump(rows, os.path.join(RES, "mask_generations.json"))
print("saved")
