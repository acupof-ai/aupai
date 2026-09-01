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
| entrypoint_help | PASS(96 个可格式化) | FAIL(fetch_corpus.py 未转义 %) | **pod 上有 main 不知道的文件**(233 个未登记 .py 之一) |
| restartability | PASS(0 new) | FAIL(attn_res_bench.py、bench_gated_mla.py 是 [NEW]) | 同上:未登记文件 |
| selftests_are_gated | PASS(47 全 gated) | FAIL(11-12 个未入 hook 表) | 同上:未登记文件带的 --selftest |
| curl_ipv4 | PASS(267 全过) | FAIL(conn3.py:12) | 同上:未登记文件 |
| no_stale_running / no_ghost_running / pod_drift / mix_supply / cited_artifacts_attested / milestone_ckpt_pinned | SKIP(无 GPU/无 /data00/无 checkpoint/非 pod) | FAIL(6 条) | **环境态**:只在 pod 存在 |

**2026-09-01 修正(tilerl 零漂移复跑,6cb62f1 之后):** 上表初版把这 4 个 FAIL 定性为"pod 文件漂移出 main"——**错了**。tilerl 把 main 全量推到 pod(273 files match,pod_drift=0,同步戳 675fb06 clean)后复跑,9 FAIL 全在,4 个仓库扫描 FAIL 一个没少。根因:**`pod_drift` 只比对已登记文件(manifest 子集),pod 上 233 个未登记 .py 对漂移检查完全隐形,但树扫描类检查(grep 整棵树)照样看得见**。`pod_drift=0` 的意思是"每个登记过的文件都一致",不是"pod 和 main 一致"。这和今晚的指纹 bug 是同一个句式:一个在空集合上成立的全称断言。

## 判决

1. **Mac 的"0 FAIL"对 6 个环境态检查是空真**——它们在 Mac 上 SKIP,不是 PASS。引用 Mac 全绿作为门禁结论是引用一个没跑的检查。(tilerl 独立撞到同一件事并已修 aded6c5:check 摘要现在自报口径——"0 FAIL of 38 run; 19 did NOT run here ... green here is not green on the pod"。)
2. **零漂移不足以让 pod 的仓库扫描 FAIL 可归因**——漂移检查的视野是 manifest 子集,pod 上有 233 个未登记文件在视野外。初版"漂移状态下 FAIL 不可归因"只对了一个方向;反过来不成立。**fb 2026-09-01 裁定的解法不是扩大漂移视野,是分层**:仓库扫描类检查的权威在 main,pod 上的扫描 FAIL 根本不归因(见判决 4)。
3. **权威 = pod 上的 drift=0 + 检查分层**(fb 2026-09-01 裁定,取代初版"两个零"):拒绝条件是 **drift≠0 一个量**——drift 比对的正是发车所切的那些文件。未登记文件数**不是**拒绝条件:229 个未登记 .py 拆开是 51 个 main 有但没推、178 个 pod 独有(只 10 个进过 git)、**168 个从没进过 git**(几个月直接写在 pod 上的一次性脚本)。pod 根目录的一次性脚本和"跑起来的那份代码是不是 main 的"无关,拿它们的计数拒绝发车是把门禁挂在 pod 的家务上。
4. **检查分层(fb 2026-09-01,同一裁定)**:仓库扫描类检查的权威在 **main**——pod 上 168 个 main 没有的文件,那些扫描在 pod 上永远不可能干净,它们的 pod FAIL 不门禁发车(读作 UNKNOWN,authority=main);环境态类检查的权威在 **pod**(缓存、显卡、账本)。Mac 跑的是 repo-scan-only,六个环境门在 Mac 上保持 UNKNOWN。实现见 scripts/launch_gate.py 的 ENV_STATE_CHECKS(客观集合 = Mac 上 SKIP 的那 19 个,selftest 守卫两边不漂移)和 _partition_fails。

## 连带发现

- **时钟倾斜污染时间比较**(fb 2026-09-01 裁定):Mac CST、pod UTC 差 8h,no_stale_running 的"27h/25h"、review_present、tasks_closed_by_commit 的时间差都不可信,按"时区未知"读;修法是全仓 time.gmtime(),fb 在加拒绝裸 strftime 的检查。
- **pod 的 6 个环境态 FAIL 里有真问题**(不归因于漂移):mix_scale_3.24b.json 的 anneal 需求 > pool 供给(6 个域全缺);milestone 3.24b 的 checkpoint 已删无钉副本(测量不可重复);no_ghost_running 检查自己连续超时(检查基础设施坏);cited_artifacts_attested 2/8 未 attest(其中 be.l1_below_constant_guess 是 44 的 facts 条目,44 自己的债)。
- **register 和 board 都是 per-worktree 文件**:worktree 里跑 `harness task`/`board.py post` 只写自己 worktree 的副本,全员不可见(3b 的 3b-3 close、44 的三条 board 帖都这么丢过)。共享副本只在集成树 /Users/bytedance/code/aupai。

## Sources

- pod 检查输出:/tmp/hcheck.txt(pod,2026-09-01,10 FAIL)
- Mac 检查输出:集成树 main HEAD,0 FAIL / 14 SKIP
- tilerl 零漂移复跑(2026-09-01):main 全量推 pod,273 files match,pod_drift=0,戳 675fb06 clean,9 FAIL 全在——4 个仓库扫描 FAIL 与漂移无关,根因是 233 个未登记 .py
- tilerl 的独立修复 aded6c5:check 摘要自报口径("0 FAIL of 38 run; 19 did NOT run here")
- runs/board.jsonl 第 [69] 行(board.py union-merge 不存在)
- scripts/harness.py(checks 注册表、SKIP 语义、pod_drift 的 manifest 子集视野)
