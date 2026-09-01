---
question: 门禁的权威在哪台机器上——为什么同一个 sha 在 Mac 报 0 FAIL、在 pod 报 8-10 FAIL
status: measured
source: fb 2026-09-01("比任何单个红项都重要");44-7 分流夜实测。checks 注册表见 scripts/harness.py
---

# 门禁权威:pod,且只在零漂移时

## 实测(2026-09-01,main HEAD)

同一棵树,两个世界:

| 检查 | Mac(main HEAD) | pod | 性质 |
|---|---|---|---|
| entrypoint_help | PASS(96 个可格式化) | FAIL(fetch_corpus.py 未转义 %) | **漂移**:pod 的 fetch_corpus.py ≠ main |
| restartability | PASS(0 new) | FAIL(attn_res_bench.py、bench_gated_mla.py 是 [NEW]) | **漂移**:pod 有 main 没有的脚本 |
| selftests_are_gated | PASS(47 全 gated) | FAIL(11 个未入 hook 表) | **漂移**:pod 文件集 ≠ main |
| curl_ipv4 | PASS(267 全过) | FAIL(conn3.py:12) | **漂移**:conn3.py 是 pod 侧文件 |
| no_stale_running / no_ghost_running / pod_drift / mix_supply / cited_artifacts_attested / milestone_ckpt_pinned | SKIP(无 GPU/无 /data00/无 checkpoint/非 pod) | FAIL(6 条) | **环境态**:只在 pod 存在 |

## 判决

1. **Mac 的"0 FAIL"对 6 个环境态检查是空真**——它们在 Mac 上 SKIP,不是 PASS。引用 Mac 全绿作为门禁结论是引用一个没跑的检查。
2. **pod 的 4 个仓库扫描 FAIL 测的是 pod 漂移,不是 main 的健康**。pod 不是 git 仓库,文件可以被任何 session 直接推改(3b 的 build_corpus.py 漂移、今晚的 fetch_corpus.py/conn3.py 都是先例),漂移状态下 FAIL 不可归因。
3. **权威 = pod,且只在 pod_drift=0 时**。修法方向(报 fb 裁定,不自己动):pod 门禁在 pod_drift≠0 时拒绝给结论(漂移让 FAIL 不可归因,和"没跑的检查不算 PASS"是同一条);Mac 的输出只能读仓库扫描那部分。

## 连带发现

- **时钟倾斜污染时间比较**(fb 2026-09-01 裁定):Mac CST、pod UTC 差 8h,no_stale_running 的"27h/25h"、review_present、tasks_closed_by_commit 的时间差都不可信,按"时区未知"读;修法是全仓 time.gmtime(),fb 在加拒绝裸 strftime 的检查。
- **pod 的 6 个环境态 FAIL 里有真问题**(不归因于漂移):mix_scale_3.24b.json 的 anneal 需求 > pool 供给(6 个域全缺);milestone 3.24b 的 checkpoint 已删无钉副本(测量不可重复);no_ghost_running 检查自己连续超时(检查基础设施坏);cited_artifacts_attested 2/8 未 attest(其中 be.l1_below_constant_guess 是 44 的 facts 条目,44 自己的债)。
- **register 和 board 都是 per-worktree 文件**:worktree 里跑 `harness task`/`board.py post` 只写自己 worktree 的副本,全员不可见(3b 的 3b-3 close、44 的三条 board 帖都这么丢过)。共享副本只在集成树 /Users/bytedance/code/aupai。

## Sources

- pod 检查输出:/tmp/hcheck.txt(pod,2026-09-01,10 FAIL)
- Mac 检查输出:集成树 main HEAD,0 FAIL / 14 SKIP
- runs/board.jsonl 第 [69] 行(board.py union-merge 不存在)
- scripts/harness.py(checks 注册表、SKIP 语义)
