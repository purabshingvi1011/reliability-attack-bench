#!/bin/bash
# Runs the benchmark for Llama-3.1-8B-Instruct in BOTH layouts (inline: documents inside the user message; tool: each document
# a tool result), five methods each, then CAG-7B (inline only), then the judge and the tables.
# 11 runs x 4,200 short replies (core suite): about 2 to 4 GPU-hours on one A100. Use --suite full to generate all 30 tests.
#
# Edit the paths below, then:  sbatch cluster/run_all.sbatch
# Check the account and partition names with Sriram (group sriramc2-ic, partition IllinoisComputes-GPU at the time of writing).
#
#SBATCH --job-name=relbench
#SBATCH --account=sriramc2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/relbench_%j.out

set -euo pipefail
BENCH_DIR=$HOME/reliability_bench                         # this folder
REPO=$HOME/belief_robustness-main                         # David's repo: only for his trained vectors and CrAM head lists
LLAMA=meta-llama/Llama-3.1-8B-Instruct                    # or a local snapshot path
CAG=ruotong-pan/CAG-7b

cd "$BENCH_DIR"; mkdir -p logs runs results
# source your environment here, e.g.:  source ~/venvs/relbench/bin/activate
# data/bench.jsonl is committed, so every run tests the same cases. It is rebuilt (from the original CrAM and CAG repos) only if
# missing; delete it first to rebuild on purpose.
if [ ! -f data/bench.jsonl ]; then
  [ -d upstream/CrAM ] || git clone --depth 1 https://github.com/Aatrox103/CrAM.git upstream/CrAM
  [ -d upstream/CAG ]  || git clone --depth 1 https://github.com/panruotong/CAG.git upstream/CAG
  python -m bench.generate --cram_repo upstream/CrAM --cag_repo upstream/CAG --suite core --out data/bench.jsonl --facts_out data/facts.jsonl
fi

# the current vectors were trained with a doubled begin-of-text token, so evaluate them the same way (drop --double_bos once
# David's retrained vectors from the simplify-data branch are in). Each layout uses the direction and the CrAM heads made for it:
# David's "inline" cell for inline, his "ipython" cell for the tool layout.
for LAYOUT in inline tool; do
  if [ $LAYOUT = inline ]; then CELL=llama31-inline; else CELL=llama31-ipython; fi
  VEC=$REPO/direction/results/models/direction_$CELL/vectors.pt
  HEADS=$REPO/cram/data/heads/heads_scores_mean_nq_$CELL.json
  COMMON="--model $LLAMA --layout $LAYOUT --double_bos"
  python -m bench.run_eval --method notag          $COMMON --out runs/${LAYOUT}_notag.jsonl
  python -m bench.run_eval --method tags           $COMMON --out runs/${LAYOUT}_tags.jsonl
  python -m bench.run_eval --method direction      $COMMON --out runs/${LAYOUT}_direction.jsonl --vectors $VEC
  python -m bench.run_eval --method direction_tags $COMMON --out runs/${LAYOUT}_direction_tags.jsonl --vectors $VEC
  python -m bench.run_eval --method direction_floor $COMMON --out runs/${LAYOUT}_direction_floor.jsonl --vectors $VEC
  python -m bench.run_eval --method direction_mean  $COMMON --out runs/${LAYOUT}_direction_mean.jsonl  --vectors $VEC
  python -m bench.run_eval --method cram           $COMMON --out runs/${LAYOUT}_cram.jsonl --heads $HEADS --bs 8
done
python -m bench.run_eval --method cag --model $CAG --layout inline --out runs/inline_cag7b.jsonl --max_new 200 --bs 8 --dtype float16

python -m bench.judge --runs "runs/*.jsonl" --model $LLAMA --out runs/judge.json
for SUB in inline tool; do
  python -m bench.score --runs "runs/${SUB}_*.jsonl" --judge runs/judge.json --out results/${SUB}_all.md
  python -m bench.score --runs "runs/${SUB}_*.jsonl" --judge runs/judge.json --fact_type fictional --out results/${SUB}_fictional.md
  python -m bench.score --runs "runs/${SUB}_*.jsonl" --judge runs/judge.json --fact_type real --out results/${SUB}_real.md
  python -m bench.score --runs "runs/${SUB}_*.jsonl" --judge runs/judge.json --source "CrAM" --out results/${SUB}_cram_data.md
  python -m bench.score --runs "runs/${SUB}_*.jsonl" --judge runs/judge.json --source "CAG NewsPolluted" --out results/${SUB}_cag_data.md
done
python -m bench.score --runs "runs/*.jsonl" --judge runs/judge.json --out results/both_layouts.md
echo ALL DONE
