"""Stage 5: activations for MASK pressure responses (teacher-forced, same features as stage 2)."""
import json, os
import numpy as np
from common import *

rows = json.load(open(os.path.join(RES, "mask_generations.json")))
tok, model = load_model()
pairs = [(chat_prompt(tok, r["pressure_messages"]), r["pressure_response"]) for r in rows]
d = extract_acts(tok, model, pairs, bs=8)
np.savez(os.path.join(ACTS, "mask_pressure.npz"), **d)
print("done")
