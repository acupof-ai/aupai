---
question: How can Claude Code sessions and the tileRL project feed reinforcement learning for the 500M code+math model, and what does each path cost?
status: recorded
source: read-only survey of /Users/bytedance/code/tilerl and this repository, 2026-09-02; file:line citations below
---

# Claude Code and tileRL as RL inputs for the 500M

## What exists, measured 2026-09-02

| asset | where | state |
|---|---|---|
| GSPO trainer, fp32 master + FP8 train + bf16 generation copies, KL anchor to the SFT reference | `algorithms/rlvr_trainer.py:81-138` | written, never run: the RL gate closed at 200M (pass@8 − pass@1 = 3.5 pt < 15 pt, `docs/lessons/reasoning_panel.md:147-154`) |
| boxed-answer reward with a ground-truth round trip (217,948 / 217,953) | `algorithms/rlvr_reward.py`, `algorithms/test_rlvr_reward_suite.py` | the only reward that has passed the round-trip rule (`docs/standards/0830v1_gates.md:488`) |
| code execution sandbox, python3 only | `datagen/sandbox_exec.run_sandboxed`, used by `eval/code_zh.py:33` | exists; no reward function wraps it |
| tool-use SFT renderer: one supervised assistant turn per pair, tool output never supervised | `scripts/loader.py:145-176` `format_agentic` | exists, no corpus has ever been packed through it |
| code-with-tests supply | `facts/data_scaling.json#ds.code_mining_feasibility` | 7.6% of repos hold impl + test, 26.96% of docs; mine, do not generate |
| tileRL GRPO on Qwen3.8-27B NVFP4, LoRA on the frozen fp4 base | `tilerl/src/tilerl/train.py:177-329` | group advantages, no PPO clip, no KL; reward = GSM8K last-number match only (`cli.py:211-216`); 27B run blocked on our eight cards (`docs/experience/wins/2026-09-02-rl-real-task.md:3-6`) |
| tileRL on-policy distillation | `tilerl/src/tilerl/train.py:337-395` | self-teacher (EMA of the adapters), plain CE on the teacher's tokens; `teacher_engine` is duck-typed (`submit`/`poll`/`step`) and nothing external implements it |
| tileRL serving | `tilerl/src/tilerl/server.py:143-251`, `generate.py` | OpenAI-compatible chat API with per-token logprobs; 8-process batch generation at 7.54× on 8 H20s |
| tileRL engine portability | `tilerl/src/tilerl/model.py:461-520`, `config.py:85-92` | hard-coded Qwen3.x hybrid loop; MLA latent KV has no field, KDA needs K = V; a 500M port means new kernels with CPU twins and gradchecks |
| Claude Code transcripts on this machine | `~/.claude/projects/*/*.jsonl` | 341 sessions, 2.69 GiB top level; 4.47 GiB with subagent sidecars; aupai + tileRL 448 MiB |

## Three paths

| path | what it is | needs GPU now | first artifact | cost | risk |
|---|---|---|---|---|---|
| A. Claude Code transcripts as agentic SFT data | parse session JSONL into ChatML turns through `format_agentic`; assistant text supervised, tool output masked; scrub paths, emails, tokens | no | a 10K-pair pilot pack with `test_sft_pack` green | days, one session | a 4096-token context holds few full tool loops; the data is one operator's style |
| B. Code-execution RL with our GSPO | reward = tests pass in the sandbox on mined impl + test pairs; the round-trip rule applies before the first RL step; gate pass@8 − pass@1 ≥ 15 pt on the SFT checkpoint | yes, after the run | reward module with its known-answer suite | reward and suite now; training after SFT | the 200M gate was flat, the 500M may be too, then RL has nothing to amplify |
| C. tileRL as teacher and judge | tileRL serves the 27B behind its API; it scores or rewrites trajectories from A, and its GRPO takes the same code reward as B through a shared verifier module | one card, after the run | a shared `verifiers` module imported by both repos | one card-day for the 27B GRPO pilot | porting the 500M into tileRL's sampler = trainer loop is not on the table: new kernels, KV pools, gradchecks |

## Ruling

A and the reward half of B start now: no GPU, both serve SFT regardless of whether RL opens. C is the joint piece with tileRL: one shared verifier module, one shared trajectory format, and tileRL's 27B GRPO pilot with the code reward on one card after the run. The 500M does not enter tileRL's engine.

Tasks: e1-24 (A), de-28 (B reward + suite), tilerl-19 (C design and the tileRL-side plugin), 44-17 folds all three into the post-pretrain plan.
