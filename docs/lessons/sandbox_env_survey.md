---
question: Which process-level sandboxes, agent-environment interfaces and reward-model-free reward schemes fit an RL loop of Claude Code + tileRL with no containers, on a k8s pod and a Mac?
status: recorded
source: web survey by a research subagent, 2026-09-02; every row carries its URL; rows marked "not verified" were not fetched that day
---

# Sandboxing, agent environments, RM-free rewards: survey for the Claude Code + tileRL loop

Constraints: no Docker (the pod is itself a k8s container; only `/usr/bin/unshare` as root, no bwrap/nsjail/firejail); macOS dev host with Seatbelt only; Python 3; hosts with no primitives must refuse or degrade explicitly.

## A. Process-level sandboxing without containers

### A.0 Two facts that decide the pod design

| fact | consequence | source |
|---|---|---|
| A k8s container with no `seccompProfile` runs `Unconfined`; `RuntimeDefault` applies the runtime profile | whether `unshare -U` works is decided by kernel + host AppArmor, not seccomp; probe, do not assume | https://kubernetes.io/docs/tutorials/security/seccomp/ |
| containerd `RuntimeDefault`: `unshare`/`setns`/`mount` need CAP_SYS_ADMIN; `clone3` → ENOSYS; `clone` only without `CLONE_NEW*` | under `RuntimeDefault` every userns tool fails EPERM; Landlock, seccomp, rlimits, setuid still work | https://raw.githubusercontent.com/containerd/containerd/main/contrib/seccomp/seccomp_default.go |

Startup probe, in-pod: `/proc/sys/user/max_user_namespaces` > 0; `sysctl kernel.apparmor_restrict_unprivileged_userns` (Ubuntu 24.04 host = 1 blocks `unshare -U`); `grep Seccomp /proc/self/status`; `unshare -Ur true`, `unshare -Urn true`, `unshare -Urmpf --mount-proc true`; Landlock via `landlock_create_ruleset(NULL,0,LANDLOCK_CREATE_RULESET_VERSION)` → ABI or ENOSYS.

### A.1 Anthropic sandbox runtime (`@anthropic-ai/sandbox-runtime`, srt) and Claude Code's sandbox

v0.0.75 (2026-09-01), Apache-2.0, Node ≥ 20.11. https://github.com/anthropic-experimental/sandbox-runtime

| question | answer |
|---|---|
| Linux primitive | bubblewrap, mandatory: `--unshare-net --ro-bind / /` + `--bind` allowWrite + `--tmpfs` denyRead, `--unshare-pid`, `--unshare-user --cap-drop ALL`; seccomp stage only blocks `socket(AF_UNIX)` |
| Linux network | empty netns; host-side HTTP + SOCKS5 proxies on Unix sockets bridged by socat to `localhost:3128/1080`; allow decision on hostname; domain fronting is an acknowledged bypass |
| macOS | `sandbox-exec` with a generated SBPL profile; network only to the proxy port |
| root | not needed (unprivileged userns) |
| without bwrap | srt refuses; Claude Code warns and runs unsandboxed unless `sandbox.failIfUnavailable: true` |
| in an unprivileged container | `enableWeakerNestedSandbox: true` binds the existing `/proc`; still needs userns |
| `claude --sandbox` flag | does not exist; the `sandbox` settings key via `--settings <file-or-json>` |
| settings keys | `enabled`, `autoAllowBashIfSandboxed`, `allowUnsandboxedCommands`, `excludedCommands`, `failIfUnavailable`, `enableWeakerNestedSandbox`, `filesystem.{denyRead,allowRead,allowWrite,denyWrite}`, `network.{allowedDomains,deniedDomains,allowUnixSockets,allowLocalBinding,httpProxyPort,socksProxyPort,tlsTerminate,strictAllowlist}`, `credentials.*`; default read policy is the whole disk including `~/.ssh` |
| headless `claude -p` | the key is loaded by every session; Agent SDK exposes the same option; scope is the Bash tool and its children only |

Sources: https://code.claude.com/docs/en/sandboxing, https://code.claude.com/docs/en/settings-reference, https://www.anthropic.com/engineering/claude-code-sandboxing, https://github.com/anthropics/claude-agent-sdk-typescript/issues/239.

### A.2 Other mechanisms

| mechanism | root? | unprivileged k8s pod | macOS | egress | note |
|---|---|---|---|---|---|
| gVisor runsc (release-20260817.0) | normally; `--rootless` needs userns | only with userns | no | `--network=none` | https://gvisor.dev/docs/user_guide/rootless/ |
| Firecracker / microVM (E2B, Vercel, microsandbox; Modal = gVisor) | needs `/dev/kvm` | no | no | vendor allowlists | https://github.com/e2b-dev/infra/blob/main/self-host.md |
| nsjail 3.6 | no if userns | iff userns | no | empty netns | https://github.com/google/nsjail |
| minijail | `-U` userns else root | iff userns | no | `-e` | https://google.github.io/minijail/minijail0.1.html |
| firejail 0.9.80 | SUID root | no | no | `--net=none` | CVE-2022-31214 |
| bubblewrap 0.12.0 | no; needs userns | iff userns | no | `--unshare-net` | https://github.com/containers/bubblewrap/releases |
| Landlock (ABI 1 = 5.13 … 4 = 6.7 TCP by port, 6 = 6.12, 8 = 7.0) | no; needs `no_new_privs` | yes regardless of seccomp/userns policy | no | TCP by port only | https://docs.kernel.org/userspace-api/landlock.html |
| Landlock from Python: `py-landlock` 0.1.1 (2026-01-11) | no | yes | no | | a ctypes shim is ~60 lines |
| seccomp-bpf from Python: `pyseccomp` 0.1.2 | no | yes under every runtime profile | no | deny `socket()` | https://github.com/cptpcrd/pyseccomp |
| `unshare -Urnmpf` (util-linux) | no if userns | iff userns | no | `-n` total | no syscall filter, no CPU/mem/fork limit, full in-ns caps |
| OpenAI Codex CLI | no | bwrap-first with `--unshare-net`; Landlock + seccomp legacy path | Seatbelt | netns off or loopback proxy | `codex debug landlock -- cmd` |
| Gemini CLI | | Docker/Podman/runsc/LXC only | Seatbelt profiles | | https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/sandbox.md |
| WebAssembly (wasmtime-py, CPython wasm32-wasi) | no | yes | yes | none | no sockets/subprocess/threads; kills pytest + numpy |
| `python -I` + rlimits | no | yes | RLIMIT_AS on macOS not verified | none | `RLIMIT_NPROC` not enforced for uid 0 |
| sandlock (multikernel, kernel ≥ 6.12) | no | yes | no | `--net-allow host:port` via seccomp notify | https://github.com/multikernel/sandlock |
| macOS `sandbox-exec` | no | n/a | yes | `(deny network*)` | deprecated in the man page, present on macOS 26; SBPL undocumented |

### A.3 Ranking for our constraints: two boundaries

**B1, the agent rollout (Claude Code's Bash tool).** Only Claude Code's built-in sandbox covers this without a container: `bubblewrap` + `socat` in the image and userns. Run with `--settings '{"sandbox":{"enabled":true,"failIfUnavailable":true,"allowUnsandboxedCommands":false,"enableWeakerNestedSandbox":true,"filesystem":{"denyRead":["~/.claude","~/.claude.json"]},"network":{"allowedDomains":["<inference-host>"],"strictAllowlist":true}}}'` plus `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`. If userns is denied the run refuses instead of running bare.

**B2, grading generated code with tests (the boundary we own; must not depend on bwrap).**

| rank | stack | needs | egress | fork bomb | secrets |
|---|---|---|---|---|---|
| 1 | `unshare -Urnmpf --mount-proc --kill-child` → drop to a non-zero uid → Landlock fs allowlist (zero TCP rules) → seccomp deny `socket`, `ptrace` → rlimits → wall-clock kill of the ns init | userns + Landlock + seccomp | total (empty netns) | `RLIMIT_NPROC` (uid ≠ 0) + pod `pids.max` | Landlock + uid + new pidns + `env -i` |
| 2 | same minus namespaces: setuid non-root → `no_new_privs` → Landlock → seccomp deny `socket` (all AF) → rlimits → subreaper kill | Landlock + seccomp | total via seccomp | `RLIMIT_NPROC` | never run tests as uid 0 |
| 3 | non-root uid + DAC (workdir owned, everything else root 0644/0600) → seccomp deny `socket` → rlimits → kill | seccomp | total | `RLIMIT_NPROC` | private `TMPDIR` in the workdir |
| 4 | nothing | | refuse; run only under an explicit `SANDBOX=none`, logged into the run record | | |

macOS: `sandbox-exec -p '(version 1)(deny default)(allow process*)(allow file-read* …)(allow file-write* (subpath "<workdir>"))(deny network*)' python -m pytest` + rlimits + process-group kill; memory unenforced.

Cross-cutting: egress control that holds is "no netns / no `socket()`"; hostname allowlists are domain-frontable. Never uid 0 for tests. Probe at startup and write the detected tier into the run record; "binary exists" is not "sandbox works".

## B. Agent RL environment frameworks (not verified by fetch that day unless a URL is primary)

| framework | env interface | Docker assumed | external OpenAI/Anthropic-compatible policy | token ids / logprobs to trainer |
|---|---|---|---|---|
| Meta OpenEnv | Gym `reset/step/state` over HTTP + WebSocket | envs as Docker images / HF Spaces | env side only | no |
| PrimeIntellect verifiers + prime-environments | `Environment` = dataset + `Rubric` + `Parser`; `SingleTurnEnv/MultiTurnEnv/ToolEnv/SandboxEnv` | no for most | yes, any OpenAI-compatible server | via vLLM in `prime-rl` |
| SWE-ReX | runtime abstraction, `LocalDeployment` bare shell | no | n/a | no |
| SWE-Gym / SWE-smith / R2E-Gym / SWE-rebench | repo@commit + F2P/P2P tests | yes for the official harness | n/a | no |
| Terminal-Bench + harbor | task dir with `Dockerfile`, `tests/test.sh` | yes | Claude Code adapter exists | no |
| Claude Agent SDK / `claude -p --output-format stream-json` | `query(prompt, options)`, hooks, `sandbox` option | no | yes via `ANTHROPIC_BASE_URL` | no; the trainer recovers them from its own server |
| OpenHands | `LocalRuntime` / `CLIRuntime` | no with local | LiteLLM | no |
| TRL GRPOTrainer | `rollout_func -> {prompt_ids, completion_ids, logprobs}` | no | rollout calls anything | yes |
| verl | `agent_loop`, token-in-token-out async server | no | internal | yes |
| slime | `--custom-generate-function-path` returning `tokens`, `loss_mask`, `reward` | no | SGLang HTTP | yes |
| NeMo-Gym | agent + environment servers, Responses-API style | no | yes | yes |
| AReaL | `RolloutWorkflow.arun_episode(engine, data)` | no | in-tree | yes |
| SkyRL | `BaseTextEnv.init/step`; HTTP endpoint for external agents; retokenization matching | no | yes | yes |
| rLLM | `BaseEnv.reset/step`, `AgentTrainer` (verl) | SWE envs only | in-tree | yes |
| Agent Lightning (Microsoft) | zero agent change: `agl` server exposes an OpenAI-compatible endpoint, traces via OpenTelemetry spans matched to served tokens | no | yes by design, OpenAI format | yes |
| ART + RULER (OpenPipe) | trajectory = messages; RULER judge | no | yes | yes |
| Atropos (Nous) | `BaseEnv.collect_trajectories/evaluate` | no | yes | yes |
| Tinker (Thinking Machines) | `Env.initial_observation/step`, `EnvGroupBuilder`, `sample()` returns tokens + logprobs, LoRA only | no | the sampling server is the service | yes, the API contract |

**Interface to mirror.** Our shape is a black-box harness rollout: setup → `claude -p` → grade. The matching designs are Agent Lightning (agent untouched, tokens recovered server-side, requests grouped by a per-rollout tag) and Tinker (`EnvGroupBuilder` = one task → G rollouts; sampler returns tokens + logprobs). Env = `(setup(workdir), prompt, grade(workdir, transcript) -> float | list[float])` with the grader as a verifiers-style rubric list; dataset = JSONL of task ids. Token recovery is tileRL's job; every `claude -p` run carries a per-rollout tag (`metadata.user_id` or a header) so tileRL groups its Messages calls into one trajectory; GRPO advantage is per episode, broadcast to all assistant turns. Do not adopt OpenEnv `reset/step`: it presumes the trainer drives steps.

**Docker-free task sets with tests (from knowledge, not verified that day).**

| set | size | licence | tests | stdlib-only | shape |
|---|---|---|---|---|---|
| HumanEval / HumanEval+ / MBPP+ (EvalPlus) | 164 / 164 / 378 | MIT / Apache-2.0 | asserts | yes | single function |
| MBPP sanitized | 427 | CC-BY-4.0 | 3 asserts | yes | single function |
| LiveCodeBench | ~1000, dated | CC-BY-4.0 (check) | stdin/stdout + functional | yes | competitive |
| APPS / CodeContests / TACO / open-r1 codeforces / PrimeIntellect verifiable-coding-problems | 10k / 13k / 25k / ~10k / 144k | MIT / Apache / Apache / CC-BY / check | stdin/stdout | yes | competitive |
| KodCode | 447k | CC-BY-NC-4.0 | pytest | mostly | single function |
| Aider polyglot (Exercism, Python subset) | 225 total | Apache-2.0 / MIT | pytest, no Docker | yes | multi-file, agentic-lite |
| Exercism python track | ~140 | MIT | pytest | yes | single module |
| BigCodeBench | 1140 | Apache-2.0 | pytest, heavy pip deps | no | function with libs |
| SWE-smith / SWE-bench lite / Commit0 / Terminal-Bench | 50k / 300 / 54 / ~90 | MIT / MIT / MIT / Apache-2.0 | pytest F2P/P2P, `tests/test.sh` | no; Docker | repo-level, agentic |

Best Docker-free agentic starting ladder: Aider polyglot Python + Exercism python + EvalPlus, pytest-graded; competitive sets for volume.

## C. Reward without a trained reward model (verified by fetch that day)

**Test-based.** DeepSeek-R1 (2501.12948): rule/test rewards only. DeepCoder-14B (together.ai blog, 2025-04): +1 only if all tests pass, no K/N credit, ≥5 tests, 6–12 s timeout. DeepSWE (2025-07): +1 only if P2P + F2P pass within 5 min, reward only on submit. UTBoost (2506.09289): 345 SWE-bench patches wrongly marked pass. ImpossibleBench (2510.20270): test editing, operator overloading, input special-casing; an "abort" option cuts cheating 54% → 9%. VeRPO (2601.03525): plain K/N biases toward easy tests. SWE-rebench V2 (2602.23866): drops tasks flaky over 50 reruns. CapCode (2606.07379): randomized tests cap the honest pass rate below 1; exceeding the cap proves cheating. Hackability audit (2606.16062): 28.5% of SWE-bench Verified accepts a wrong patch.

**Judge inside GRPO groups.** J1 (2505.10320): both orderings, reward consistency. RULER (openpipe.ai/blog/ruler, 2025-07): frozen LLM ranks all N group rollouts in one call; only relative scores matter under GRPO. Rubrics as Rewards (2507.17746) and Checklists (2507.18624): frozen judge over instance rubrics. Pref-GRPO (2508.20751): within-group pairwise win rate. ArenaRL (2601.06487): seeded tournament, O(N) judge calls. Position bias survives rubrics (2602.02219). Rubric Dropout (2608.11669): training-judge score rises while the gold judge falls 22 pt.

**Intrinsic / self-play.** TTRL, entropy minimisation, Intuitor, spurious rewards (Qwen contamination artefact), SRT collapse, CURE, R-Zero: single-signal methods rise then collapse; none shown for multi-turn agentic coding; self-play needs an executor as ground truth.

**Failure → mitigation.**

| failure | mitigation |
|---|---|
| editing/deleting/skipping tests, `exit(0)`, monkeypatching pytest | tests live outside the agent's writable tree and are copied in fresh at grade time onto a clean checkout with only the `src` diff applied; reject diffs touching test paths; parse pytest's summary, not the exit code |
| input special-casing | hidden tests; ≥5 tests; randomised inputs |
| partial-credit gaming | binary all-pass, or difficulty-weighted |
| flaky tests / timeouts | N reruns at curation; timeout = fail |
| judge position bias | both orderings, require agreement |
| verbosity / self-preference | rubric anchoring; judge from another model family |
| non-transitivity | whole-group ranking in one call, or tournament |
| judge drift | frozen stronger judge, periodic gold audits |
| monitor in the reward | keep any hack monitor for logging and filtering only |

Composition for us: tests first, binary; judge only to break ties inside the all-pass or all-fail subgroup; judge = frozen stronger model of another family, both orderings, rubric anchored on the task spec; never judge-only when tests exist.

## Recommendation

Two boundaries, both probed at startup with the detected tier written into every run record. B1: install `bubblewrap` + `socat` in the pod image and enable Claude Code's built-in sandbox through `--settings` with `failIfUnavailable`, `enableWeakerNestedSandbox`, `denyRead` on `~/.claude`, `allowedDomains` limited to the tileRL host; on macOS the same key uses Seatbelt. B2: a ~200-line Python wrapper we own: `unshare -Urnmpf` when userns is permitted, then drop to a non-zero uid, Landlock fs allowlist, seccomp deny `socket`/`ptrace`/`setsid`, rlimits, wall-clock kill; fall back to uid + Landlock + seccomp, then uid + DAC + seccomp, then refuse. Mirror Agent Lightning/Tinker for the environment; tileRL logs `(request_id, prompt_ids, completion_ids, logprobs)` per Messages call and groups by a per-rollout tag. Start with Aider polyglot Python + Exercism + EvalPlus, grade on a fresh checkout with hidden tests copied in at grade time, binary all-pass, timeout = fail; a frozen cross-family judge with both orderings only to break ties within a group.
