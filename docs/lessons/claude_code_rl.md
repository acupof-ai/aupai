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

## Ruling (user, 2026-09-02 16:05 +0800: build the basic adaptation first)

The loop is: **Claude Code is the agent harness, tileRL is the inference engine and the trainer, a
container is the environment, tests are the reward.** The policy is whatever tileRL serves; the
27B first, the 500M when it can drive a tool loop. The 500M does not enter tileRL's engine as a
model class; it enters the loop as a served policy behind the same API when it is ready.

| stage | deliverable | GPU | owner | done when |
|---|---|---|---|---|
| 1. API adaptation | an Anthropic Messages-compatible shim in front of tileRL's OpenAI server: messages, system, tools, `tool_use`/`tool_result` blocks, streaming, `stop_reason`; per request it logs request id, prompt and completion token ids, per-token logprobs; Claude Code points at it through `ANTHROPIC_BASE_URL` | none: tileRL's `tiny` model on CPU | tilerl | `claude -p "<task>" --output-format stream-json` completes one tool call and one final answer against the tiny model |
| 2. Launcher, no container | one command starts server + shim + `claude -p` on a task checkout; **no Docker**: the pod is already a container and nested containers are not available there, and the launcher must also run on a bare host | none | tilerl | cold start to a finished rollout on the Mac and on the pod with the same command |
| 3. Isolation per rollout | a temporary git worktree per rollout plus a process sandbox from an existing open-source tool, chosen by what the host offers: levels measured 2026-09-02: `bwrap`/`nsjail`/`firejail` where present (none on the pod or the Mac) > `datagen/sandbox_exec` (pod: root + `unshare`, chroot + namespaces + rlimits) > macOS Seatbelt `sandbox-exec` (Mac: the only isolation the host offers; deprecated but in use by browsers) > rlimits + timeout + private tmpdir, which **refuses to execute** unless `ALLOW_UNISOLATED=1`; network off except the shim; K rollouts in parallel; the same code path reports which level of isolation it got | none | de | K=8 rollouts of one task on the pod (no container available) and on the Mac, diffs and test results collected, isolation level recorded per rollout |
| 4. Reward, no trained reward model | (a) tests pass when the task ships tests (`(code, tests) -> float`, round-trip rule); (b) otherwise the 27B served by tileRL judges the K rollouts of one task **pairwise within the group**, and only the group-relative order is used; (c) GRPO group-normalised advantages, optional KL to the reference; no critic is trained. Trajectory = shim token ids + logprobs aligned to the transcript by request id | none for (a); the serving card for (b) | de (a), tilerl (b) | one rollout's shim record equals the sampled sequence by `torch.equal`; judge order agrees with test order on 20 tasks that have both |
| 5. Training | GRPO in tileRL over K rollouts per task with the stage-4 rewards, LoRA on the served weights, sampler = trainer so no weight sync | one card after the run | tilerl | reward rises on a 20-task pilot and MMLU does not drop within noise |

Order is fixed: no stage starts before the previous one's "done when" is an artifact. Stages 1-3 and 4(a) need no GPU and start now. User constraints 2026-09-02 16:25 +0800: no containers (some environments have none), open-source sandboxing, no trained reward model. Tasks: tilerl-19 covers stages 1, 2, 4(b) and 5; de-28 stage 3 and 4(a); e1-24 the trajectory alignment.
