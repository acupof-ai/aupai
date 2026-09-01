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

## 边界条件:上限是回溯性的,不是原理性的(de 2026-09-01,fb 裁定)

判决 3-4 把假红归给"pod 副本 vs main"这个轴。有一类假红不在这个轴上,而它的上限只对**已经产生的产物**成立。

实例是 `22de0e7`(「take main's gate_corpora fingerprint fix over mine」):双方各自独立修了同一个 bug,合并取 main 那版,语义上什么都没丢,`merge_complete` 报 FAIL。它是纯 git 历史问题,在 main 上、在任何 worktree 上、在 pod 上跑结果都一样——没有哪棵树的权威能改变它。原因是两种粒度各有一个反例,两个都实测在案:

| 判据粒度 | `f981b21`(真丢弃 e1 的 `_here`) | `22de0e7`(取一侧正确) | `_broken_merge_complete()`(真丢弃 `OURS_MARKER`) |
|---|---|---|---|
| top-level `def` 名 | FAIL ✓ | PASS ✓ | **PASS ✗** |
| 代码行(去注释,按集合) | FAIL ✓ | **FAIL ✗** | FAIL ✓ |

- def 粒度漏掉破坏世界:丢的 `OURS_MARKER` 是**活函数里的一行**,签名还在,def 级看不见。
- 行粒度误报 `22de0e7`:它有 **18 行**代码合法地不在结果里(自己那版等价修法及其注释),没有一行是损失。

所以 `_content_restored` 的行粒度**只在已被判定 contested 的单个路径上安全**,提到全 merge 扫描必然二选一地错。

**关键是这个上限的性质(fb 裁定):它是回溯性的,不是原理性的。** 一个没有记录自己产生时代码身份的产物,事后无法被接回历史——这一点不可修复。但一个记录了的产物可以,而记录是事前一行的事。所以正确的结论不是"放弃这类问题",是"事前记,事后就是一次 join"。

事前机制已经有一半:`scripts/pod_push.sh:31` 的 `stamp_sync` 在每次全量推送时把 `git rev-parse HEAD` 写进 pod 的 `data/pod_synced_head`(实测 2026-09-01:`cbee9ca2732f87aa186e72d6b982f7dc835e329b 0 2026-09-01T15:50:58Z`,dirty=0),而按名推送会**清除**戳而不是声称一个 sha——因为那时 pod 是"一个 sha 的树加另一个 sha 的一个文件",诚实状态是 unknown。e1 的 `exp.py --commit` 是另一半。

**缺的那一半在语料侧,而且比预想的更缺。** 实测 pod 上 43 个 `build_corpus_stats.json`:**没有一个带代码 sha**,`filters_fp` 只有 14 个带,`fingerprint` 43 个全带。所以"这批语料是哪个版本的代码建的"目前只能靠人翻 commit——`df2b774` 那五条中文正则从未到达 pod、九域在它不存在的情况下建成,就是靠翻当时的 manifest 才定案的。下一次构建应把 `pod_synced_head` 的 sha 写进 `build_corpus_stats.json`,这个问题从此自动可答。

**这一节的两个用途**:防止第三次尝试粒度(两个方向的墙都在上表里,要动先给出同时通过三列的判据);以及说明为什么"给产物加一行身份"值得做——它把一整类事后无法回答的问题变成 join。

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
- 边界条件一节的三列:`f981b21` / `22de0e7` / `_broken_merge_complete()`,de 2026-09-01 实测,判据见 `scripts/harness.py` 的 `_content_restored`;`merge_complete` 逃生口修复 accc868
- 事前身份:`scripts/pod_push.sh:31` `stamp_sync`;pod 上 `data/pod_synced_head` = `cbee9ca2 0 2026-09-01T15:50:58Z`(de 实测);pod 上 43 个 `build_corpus_stats.json` 的键频:`fingerprint` 43、`filters_fp` 14、代码 sha **0**
