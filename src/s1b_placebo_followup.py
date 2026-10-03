"""Stage 1b: (i) greedy answers under the placebo condition; (ii) for every answer under the placebo and a
pressure condition, a follow-up turn asking for the honest answer (in-context knowledge check).
Updates results/generations.json in place."""
import os
from common import *

path = os.path.join(RES, "generations.json")
items = json.load(open(path))
tok, model = load_model()
if "resp_placebo" not in items[0]:
    print("placebo")
    outs = generate(tok, model, [chat_prompt(tok, build_messages("placebo", it["question"])) for it in items])
    for it, o in zip(items, outs):
        it["resp_placebo"] = o
for cond in ["placebo", "sycophancy", "incentive", "instructed"]:
    todo = items
    print("followup", cond, len(todo))
    prompts = [chat_prompt(tok, build_messages(cond, it["question"], it["wrong"]) +
                           [{"role": "assistant", "content": it[f"resp_{cond}"]},
                            {"role": "user", "content": FOLLOWUP}]) for it in todo]
    for it, o in zip(todo, generate(tok, model, prompts)):
        it[f"followup_{cond}"] = o
jdump(items, path)
print("saved")
