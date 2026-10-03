"""Stage 0: download datasets into data/ (TriviaQA rc.nocontext validation, MASK subsets, probe training sets)."""
import os, urllib.request
from datasets import load_dataset
from common import DATA

os.makedirs(DATA, exist_ok=True)
load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").to_parquet(os.path.join(DATA, "triviaqa_val.parquet"))
for c in ["known_facts", "doubling_down_known_facts", "disinformation", "continuations"]:
    load_dataset("cais/MASK", c, split="test", token=os.environ.get("HF_TOKEN")).to_parquet(os.path.join(DATA, f"mask_{c}.parquet"))
for url, name in [
    ("https://raw.githubusercontent.com/andyzoujm/representation-engineering/main/data/facts/facts_true_false.csv", "facts_true_false.csv"),
    ("https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv", "cities.csv")]:
    urllib.request.urlretrieve(url, os.path.join(DATA, name))
print("data ready")
