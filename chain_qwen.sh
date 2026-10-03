#!/bin/bash
# Replication of the TriviaQA experiment on Qwen2.5-7B-Instruct (same questions, same prompts, same judge).
cd "$(dirname "$0")/src"
export HF_HOME=$(cd .. && pwd)/hf_cache RUN_TAG=qwen7b MODEL_ID=Qwen/Qwen2.5-7B-Instruct
PY=../.venv/bin/python
mkdir -p ../logs/qwen7b
[ -f ../results/qwen7b/generations.json ] || N_Q=4000 $PY s1_generate.py > ../logs/qwen7b/s1.log 2>&1
$PY s1b_placebo_followup.py > ../logs/qwen7b/s1b.log 2>&1 && $PY s2_acts.py > ../logs/qwen7b/s2.log 2>&1 && \
$PY s1c_judge_tqa.py > ../logs/qwen7b/s1c.log 2>&1 && $PY s6_analysis.py > ../logs/qwen7b/s6.log 2>&1 && \
$PY s6b_stats.py > ../logs/qwen7b/s6b.log 2>&1 && $PY s8_figures.py > ../logs/qwen7b/s8.log 2>&1
echo QWEN_DONE $? > ../logs/qwen7b/done
