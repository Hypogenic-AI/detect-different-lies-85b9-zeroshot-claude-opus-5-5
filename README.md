# Lies or hallucinations? A matched knowledge × incentive study of false answers

**Question.** When an LLM says something false, is it because it does not know the answer (hallucination) or because it knows the answer and misreports it under an incentive (lie)? And do standard white-box lie detectors and hallucination detectors tell these two apart, or do they just respond to falsehood, uncertainty or the prompt?

**Paper:** [`paper_draft/main.pdf`](paper_draft/main.pdf) (LaTeX source: `paper_draft/main.tex`).

## Design (one model, matched questions)
- **Model:** Llama-3.1-8B-Instruct (main). Qwen2.5-7B-Instruct is a replication of the TriviaQA part.
- **Questions:** 4,000 TriviaQA (`rc.nocontext` validation) questions with short answers.
- **Knowledge factor:** under a neutral prompt we take 1 greedy answer and 10 answers sampled at T=1.
  - **Known**: greedy correct and ≥80% of samples correct.
  - **Unknown**: greedy wrong, ≤20% of samples correct, and the samples are inconsistent.
  - **Confident misbelief**: consistently wrong.
  - **Partial**: everything else.
- **Incentive factor:** five conditions on the same questions.
  - `neutral`.
  - `placebo`: TV-quiz role, no incentive.
  - `sycophancy`: the user bet $100 on a wrong answer.
  - `incentive`: the company loses $10k per correct answer; no instruction to lie.
  - `instructed`: an explicit "give a false answer".
- **Labels:**
  - **Lie** = a false answer to a Known question. **Hallucination** = a false answer to an Unknown question. Both are compared *under the same prompt*.
  - A follow-up turn ("honestly, what is the correct answer?") checks in context whether the model knew.
  - Grading uses a local LLM judge (Qwen2.5-14B-Instruct; 97.8% agreement with 90 held-out hand labels). Substring alias matching fails on deceptive answers such as "Gordon Ramsay's brother".
- **Detectors** (layer 16, mean over answer tokens; trained on 50% of questions and tested on the rest):
  - off-policy RepE/Goldowsky-Dill deception probe;
  - Marks & Tegmark-style truth probe;
  - instructed-lie probe (cross-prompt);
  - within-prompt lie probe;
  - SEP-style hallucination probe;
  - answer NLL in its own context and in the neutral context;
  - a new *knowledge-gated falsity* score;
  - a supervised lie-vs-hallucination probe.
- **MASK:** 630 pressure scenarios (known_facts, doubling_down, disinformation, continuations), each with MASK belief elicitation.

## Main findings (Llama-3.1-8B-Instruct)
1. **Most false answers are epistemic.** The share of false answers that come from Known questions (i.e. are lies) is 4.9% under the placebo prompt, **6.7% under the implicit incentive**, 24% under sycophancy and 44% under explicit instruction.
   - The implicit incentive raises P(false | Known) from 2.3% to only 3.2%. The difference is statistically significant but tiny (+0.9 pp, McNemar p = 0.0015).
   - Under instruction, 89% of lies are retracted in the honest follow-up. For Unknown questions, only 4–9% are corrected.
2. **The answer depends on the question distribution.** On MASK the composition reverses: 64% lies, 34% no consistent belief, 2% honest errors.
3. **Standard detectors do not separate lies from hallucinations.** With both answers false and under the same prompt, the deception, truth, instructed-lie, within-prompt-lie and hallucination probes reach AUROC 0.24–0.59 for sycophancy and incentive. Several rank hallucinations as more suspicious than lies. In-distribution lie probes turn into falsity detectors: they separate hallucinations from correct answers at 0.86–0.94.
4. **Prompt confound.** The deception probe separates truthful answers given under the lie instruction from neutral truthful answers at AUROC 1.00, and at 1% FPR it flags 100% of them. The instructed-lie probe reaches 1.00 from the last prompt token alone, before any answer exists.
5. **What separates lies from hallucinations is knowledge.**
   - A supervised lie-vs-hallucination probe reaches 0.79–0.89. It cannot separate lies from correct answers (0.48–0.58) and already works from the prompt token (0.81–0.88).
   - Combining falsity with pre-answer knowledge (*knowledge-gated falsity*, no lie-vs-hallucination labels) ranks lies above hallucinations at 0.64–0.72.
6. **On MASK**, the deception probe separates lies from honest-correct answers at 0.76. This drops to 0.62 within the most natural subset (known_facts), and there are too few honest errors (n = 9) to test lies vs errors.
7. **Qwen2.5-7B-Instruct replication** (same questions and pipeline; `results/qwen7b/`, paper §4.6).
   - The implicit incentive again has no effect: P(false | Known) is 4.5% vs 4.1% under the placebo (p = 0.28).
   - Sycophancy is stronger than in Llama (35% of Known questions), while explicit instruction is mostly refused (11%).
   - On same-prompt lie vs hallucination, standard probes reach 0.23–0.56. The deception probe again reads the prompt (0.94 on answer tokens, 0.99 from the prompt token alone).
   - The supervised lie-vs-hallucination probe is again a knowledge probe: 0.75–0.81 on its own task, 0.29–0.47 on lie vs correct.

## Layout
```
src/                 pipeline (stage order below), common.py = prompts, grading, model I/O
results/             raw generations, judge labels, labels.csv, AUROC / flag-rate / confound CSVs, JSON summaries
results/qwen7b/      same for the Qwen2.5-7B replication
results/tqa_judge_v1_strict.json  first (rejected) judge prompt run, kept for transparency; superseded by tqa_judge.json
results/previews/    PNG previews of figures
paper_draft/         main.tex, references.bib, figures/, tables/ (auto-generated by s8_figures.py)
logs/                stage logs
data/, hf_cache/     datasets / model weights (git-ignored); results/acts/ = activations (git-ignored, ~2 GB)
```

## Reproduce
Requirements: one GPU with at least 40 GB (A6000 used), plus HF_TOKEN with access to `meta-llama/Llama-3.1-8B-Instruct` and `cais/MASK`.
```bash
export UV_CACHE_DIR=$PWD/.uvcache UV_PYTHON_INSTALL_DIR=$PWD/.uvpy
uv venv .venv -p 3.11
uv pip install -p .venv torch==2.7.1 transformers==4.56.2 accelerate "datasets>=3" pandas scikit-learn matplotlib scipy pymupdf
cd src && export HF_HOME=$PWD/../hf_cache
python s0_data.py                  # TriviaQA, MASK, RepE facts, cities -> data/
N_Q=4000 python s1_generate.py     # neutral greedy + 10 samples, distractors, sycophancy/incentive/instructed answers (~1.5 h)
python s1b_placebo_followup.py     # placebo answers + honest follow-up turns (~35 min)
python s2_acts.py                  # teacher-forced activations (layers 8-28), NLLs, probe training sets (~15 min)
python s1c_judge_tqa.py            # Qwen2.5-14B judge for all answers/samples/follow-ups (~1 h)
python s1c_judge_tqa.py validate2  # judge vs held-out hand labels
python s3_mask_generate.py && python s5_mask_acts.py && python s4_mask_judge.py   # MASK (~1 h)
python s6_analysis.py              # labels, composition, detectors, AUROCs, confound, layer sweep (~20 min CPU)
python s6b_stats.py                # McNemar tests, bootstrap CIs
python s7_mask_analysis.py && python s7b_mask_subsets.py
python s8_figures.py               # figures + LaTeX tables
cd ../paper_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main
# Qwen replication: ./chain_qwen.sh  (sets RUN_TAG=qwen7b MODEL_ID=Qwen/Qwen2.5-7B-Instruct)
```
Seeds are fixed: question sample `random_state=0` and train/test split `default_rng(0)`. Sampling at T=1 is stochastic on GPU, so knowledge labels may vary slightly between reruns.

## Notes and limitations
- No external LLM API was usable (the OpenRouter quota was exhausted and the OpenAI key was invalid), so all judging is local.
- 7–8B models rarely lie under implicit incentives. Most lies here come from sycophancy and explicit instruction.
- Knowledge labels come from sampling consistency and are therefore noisy.
- See the paper's limitations section for more.
