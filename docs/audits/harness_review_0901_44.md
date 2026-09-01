---
question: harness check 框架本身的跨审——CHECKS 注册表/selftest 机制/坏世界构造/SKIP 语义/cfg_default/规则覆盖表诚实性,51 个检查单元里有没有原则级缺陷存活
status: recorded
source: fb tasking 2026-09-01 09:xx (44-5); scripts/harness.py @ 8151 行; 实证复现 exit -14
---

# Harness check framework cross-review (44-5)

**Verdict: 两个原则级缺陷存活。** (1) 每个 check 的 5s 超时不是 SKIP,是 SIGALRM 默认处置杀死整个 harness 进程(exit -14),不留 check 名字——fb 今晨看到的 "check run itself failed, exit -14" 的根因,已实证复现。(2) 规则覆盖表有三行伪造映射,把规则挂到不执行它的 check 上。另有 20 个 defect、若干 nit。框架的骨架(坏世界元检查、空转元检查、字面量棘轮、真工件变异)是同类实现里我见过最好的,但超时路径和覆盖表诚实性是原则级的。

Scope: CHECKS 注册表 43 条 + 16 个动词 selftest(_selftest_*,5328-5896、6325-6340 调用)= 59 个测试单元全扫;框架机制(run_checks、_demo、SKIP 语义、cfg_default、_RULE_CHECKS/_MANUAL_RULES、pre-commit hook)。方法:读全部 check 函数 + 实证(复现 -14、证无 signal.signal、跑 is_pod 判定)。

## Headline 1 — 超时即无名死亡(P3+P2,原则级)

`run_checks` (scripts/harness.py:4827) 对每个 check 设 `signal.alarm(5)`,但**全文件没有任何 `signal.signal(SIGALRM, handler)`**(grep 实证:0 处)。SIGALRM 默认处置是终止进程。4829 行的 `except TimeoutError` 是死代码——没有任何 handler 抛 TimeoutError。42-43 行注释声称 "a timed-out check SKIPs and names itself in the output"——从未发生。

实证(本审复现):

```
$ python3 -c 'import signal,time; signal.alarm(1); time.sleep(3)'  → exit -14
# harness 内复现:把第一个 check 换成 sleep(8) 跑 run_checks → exit -14, stdout/stderr 全空
```

任何 check 超 5s(模板扫描预算 90s)→ 整个 harness 死亡,exit -14,hook 打印 "no [FAIL] line -- the check run itself failed, exit -14",不说哪个 check 超时。这正是 fb 今晨遇到的拒绝。模板扫描在全数据 checkout 上实测 27s,离 90s 预算有 3.3x 余量——但数据再涨 3.3 倍、或任何 check 回归到 >5s,就是无名死亡。

Fix(三层,按序):
1. 模块加载时装一次 handler:`signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))`(主线程;run_checks 只在主线程跑)。
2. 每个 check 跑之前向 stderr 打名字(flush)——死亡时 hook 的 stderr tail(hook 已有此逻辑)能带出最后一个 check。
3. 超时后的 SKIP 在 hook 里应视为 FAIL 或至少被 hook 打印:hook 成功时只打印 [FAIL] 行,SKIP 行不可见——一个持续超时的 check 等于永久静默放行(见 D5)。

## Headline 2 — 规则覆盖表的伪造映射(P2/诚实性,原则级)

`agents_rules_covered` 只验证"映射存在"(424-425 行 docstring 自认 "Coverage cannot prove a mapping is honest"),所以 AGENTS.md 的规则覆盖表是唯一的诚实闸门——它有三行是错的:

| AGENTS.md 规则 | 表/代码挂的 check | 实际执行者 | 问题 |
|---|---|---|---|
| `CUDA_VISIBLE_DEVICES, not cuda:N` (AGENTS.md:271; harness.py:110) | `gemm_dims_aligned` | `device_set_honoured` | gemm_dims 对齐 vocab/d/ffn_hidden 的 8/16 整除,与设备索引毫无关系;真正执行的 check 没挂任何规则 |
| `runs/.jsonl ledgers merge by union` (AGENTS.md:282; harness.py:113) | `no_ghost_running` | `ledgers_one_line_per_row` | ghost_running 查 running 行有无活进程;union 行格式由 ledgers_one_line_per_row 执行,后者没挂任何规则 |
| `CI gates` (AGENTS.md:265; harness.py:107) | `"CI"` | 无 | "CI" 不是 check 名;431 行的 known 集合把它白名单化,映射不可证伪 |

读者信任 "CUDA_VISIBLE_DEVICES 规则由 gemm_dims_aligned 执行" 就会停止寻找——而 device_set_honoured 确实存在、确实在跑,只是没挂到规则上。另:`8×H20, all usable → pod_drift`(harness.py:118)也是松映射(pod_drift 查文件漂移,不查 GPU 数量/可用性)。

Fix:三行改挂真实 check;给 `agents_rules_covered` 加一条 selftest——每个被映射的 check,其 asserts 字符串与规则文本有词面交集(拦不住刻意伪造,但拦得住张冠李戴)。

## Defect 级发现

| id | file:line | principle | failing case / artifact | sev | fix |
|---|---|---|---|---|---|
| D1 | harness.py:1480 | P2+P7 | `pgrep -f <name>` 匹配任何 cmdline 提及 run 名的进程——已实证的 live case(milestone watcher 提及 run 名 → 过期 running 行不可见)。另:1465-1470 行**裸读** ledger,不经 `_exp_events` fold——append-only close 后 running 行仍在,进程退出后 pgrep 无命中 → 正确关闭的 run 2h 后读成 ghost(no_stale_running 已用 fold,此 check 没有;pod 侧待确认) | defect | de 已在做 PID 文件方案(runs/<name>.pid);裸读改 `_exp_events(root)` |
| D2 | harness.py:474-515 | P2+P6 | `grep -E 'train[.]py\|run_ddp'` 匹配任何提及 train.py 的 cmdline(`tail -f train.py.log`、`vim`);leaders 集合从**过滤后**的行算(498 行滤掉 setsid wrapper,511 行 leaders 不含它 → 它的子进程读成 orphan)。496-506 行记录了 2026-09-01 两次对正确启动的任务的误拒。坏世界是 SelftestSkip(518-524)→ **此 check 没有失败用例**,selftest 从不执行它 | defect | PID 文件 + 从 launcher 自己的 pid 文件取 session 几何(P7);补一个可构造的坏世界 |
| D3 | harness.py:322 | P2+P5 | pin = 任何匹配 `*milestone_{tok}*.pt` 的**文件名**;内容从不验证。一个空文件 `milestone_3.24b.pt` 即满足。3.24B 丢失事故的本质是字节没了,此 check 验证的是名字 | defect | pin 携带 ckpt 的 sha256+size,check 比对 |
| D4 | harness.py:2006-2024, 1971-1999 | P5 | 模板扫描缓存:key 不含扫描代码(靠手 bump `_TEMPLATE_CACHE_VERSION`,忘 bump = 旧判决静默服役);大 SFT 源只 hash 头+尾 64KB,**同尺寸中间编辑**不可见——1974-1975 行注释 "Head+tail catches same-size edits" 对中间编辑是假的 | defect | key 加 `inspect.getsource(_template_scan)` 的 hash;源文件改分块采样 hash 或承认上限 |
| D5 | harness.py:4830 + hook | P3 | 超时 → SKIP → exit 0 → hook 放行。hook 成功时不打印 SKIP 行,持续超时的 check 永久静默不守门。注释的理由(逼出 --no-verify)针对的是挂死;**具名**的超时 FAIL 是可行动的,无名死亡不是 | defect | 超时 SKIP 在 hook 中按 FAIL 处理,或 hook 打印 SKIP 行 |
| D6 | harness.py:3000-3041 | P2+P6 | score_matrix:(a) fold 是位置最后赢,无 terminal-beats-running(`_exp_events`:1425 已有正确规则,此 check 重造了一个弱版)——ok 后混入一条 running 事件就掩盖未打分的 ok run;(b) 豁免 token 硬编码("train.py","sft_math","rlvr","run_ddp.sh","run_sft.sh"),新 launcher 天然豁免;(c) `"--profile" in cmd` 命中 `--profile_steps`——带 profile flag 的真训练永久豁免;(d) PASS 证据无计数,0 个 ok run → 空转 PASS 且元检查看不见 | defect | 用 `_exp_events`;豁免改从注册表派生;`--profile` 改词边界;证据带 ok-run 计数 |
| D7 | harness.py:3125-3126, 3169-3170 | P3 | env_fp/opt_state:`except Exception: continue`——**损坏的 checkpoint 静默豁免**,且没有任何 check 负责 checkpoint 可读性 | defect | 不可读 → FAIL 或 WARN |
| D8 | harness.py:1718-1719 | P3 | sft_pack_holdout:`pack_fp is None` → PASS "UNKNOWN, not verified"。此 check 存在的理由就是防旧包,旧包恰好没有 fp——精确地在要抓的形状上放行 | defect | 至少 WARN(对照 3179 行 opt_state 的诚实写法 "0/N -- not a live PASS") |
| D9 | harness.py:1667-1677 | P3 | score_input_fresh:1 个 fresh + N 个 unrecorded → PASS 带注记。asserts 是 "a score records which corpus it scored",未验证的域违反 asserts | defect | unrecorded > 0 → WARN 或 FAIL |
| D10 | harness.py:527-551 | P2+P6 | curl_ipv4:文本正则代理;根目录 train.py 不扫(只 5 个子目录);PASS 证据无计数(0 处 curl → 空转 PASS,元检查看不见);`"""[\s\S]*?"""` 剥注释可能过剥 | defect | 扫根级 .py;证据带命中计数 |
| D11 | harness.py:985-1003 | P2 | pinned_ids:asserts 说 "four files hardcode these ids",只查 loader.EOS_ID + Cfg.num_id 两个 | defect | asserts 改两个,或补齐另两个文件 |
| D12 | harness.py:864 | P2 | mix_not_unfiltered:`"web" in doms`——未过滤语料换个域名(web_raw/cc_unfiltered)就过 | defect | 按语料指纹或规模特征判,不按名字 |
| D13 | harness.py:1341 | P2 | entrypoint_help:只扫 5 个子目录的**顶层** *.py;train.py(根目录,有 argparse)和子目录 CLI 全在范围外 | defect | 递归 + 根级 |
| D14 | harness.py:1516-1522 | P2 | gemm_dims:只读整数字面量赋值;`vocab = BASE + 73` 这类计算维度静默不查;只有三个全非字面量才 FAIL | defect | 字面量求值失败的字段 → WARN 列出 |
| D15 | harness.py:3361 | P2 | frozen_keys:`src.index("ArgumentParser")`/`src.index("parse_args")` 取首次出现——注释里提到 parse_args 就截断区段 → 漏掉新 flag → check 在不全的集合上通过 | defect | 用 ast 定位 argparse 区段 |
| D16 | harness.py:3428-3511, 3531 | P2+P3 | mix_supply:val 切分算术重造 train.py:1583,无同步测试;`HARNESS_TOKEN_CACHE_DIR` 在坏世界设置后**从不 pop**(6039 行清理只 pop REQUIRE_EXTRA + GPU_PRESENT)→ 泄漏进空转循环和后续 selftest | defect | 算术抽成 train.py 可导入的函数;清理列表补全 |
| D17 | harness.py:4085-4100, 4210-4211 | P2+P3 | lane:`HARNESS_BUSY_CARDS`/`HARNESS_TRAINING_PROC` 同样从不 pop;`ps aux` 子串匹配——`vim train.py` 读成训练在跑 → 小任务占卡时 PASS | defect | env 清理;训练进程判定改 PID 文件/cmdline 精确匹配 |
| D18 | harness.py:4044-4045 | P3 | tasks_stale:opened 日期不可解析 → `pass` 静默跳过;no_stale_running 同样形状是 FAIL(1442-1444)。不一致的 fail 方向 | defect | 对齐为 FAIL |
| D19 | harness.py:436 | P2 | agents_rules_covered:`nb.startswith(_norm_rule(k)[:38])`——新 bullet 与已有 key 共享 38 字符前缀即被静默"覆盖";两个 key 共享前缀时映射歧义 | defect | 前缀匹配改全串归一化相等,或要求唯一前缀 |
| D20 | harness.py:3679-3686 | P2 | env_importable:_REQUIRED 手工维护——新增第三方 import 不进表就不查,而这正是本 check 防的事故类(容器重启丢了表外的手装包) | defect | 从 import 扫描派生,或把缺口写进 asserts |

## Nit 级

| id | file:line | 问题 |
|---|---|---|
| N1 | harness.py:405,459,558,2321,2381,2394,2400,4000,4058 等 ~10 处 | 坏世界契约不一致:SelftestSkip(响亮跳过) vs `return None`(在 6023 行 os.walk(None) 崩,被 6035 行抓成 "raised instead of reporting FAIL",selftest 以混乱消息失败)。统一为 SelftestSkip |
| N2 | harness.py:2316-2342 | review_present 坏世界删 review[0];若该任务关闭 <30min,check 报 WARN 不是 FAIL → selftest 成败取决于**何时跑**。坏世界里把 closed 改老 |
| N3 | harness.py:3688 | `_LINUX_ONLY` 定义后零引用,死代码 |
| N4 | harness.py:3796-3816, 3926-3948 | pipeline 路径 `_task_open_run`/`_task_close_run` 用 `_write_tasks` 全量重写,cmd_task 用 `_append_task` 追加——违反 3760-3768 自己的事件日志契约;重写丢历史事件且与 union merge 竞速 |
| N5 | harness.py:2562-2582 | entrypoints_ran:任务 token ≥5 字符模糊匹配("train"/"eval" 几乎万能)→ 没跑过的脚本可计为 tried+ok;`matched[-1]` 是文件位置最新非时间最新 |
| N6 | harness.py:2597 | entrypoints_table_present 数的是**所有**引用脚本的表格行,不是入口表;删了入口表但别的表引用脚本仍 PASS(坏世界自认:删全行才 FAIL) |
| N7 | harness.py:2933-2943 | readme_current 坏世界同时触发 (a) 目标词缺失和 (b) 退休短语——(b) 未隔离;(a) 逻辑坏了世界仍 FAIL |
| N8 | harness.py:3545-3553 | corpus_fp:PROVENANCE 标题含两个域 token 时,一个指纹同时归属两域 |
| N9 | harness.py:1235 | merge_complete PASS 证据的 contested 计数用**未过滤**的第三次调用,把 .jsonl 和 0-commit 行也算进去,数字偏大 |
| N10 | harness.py:6088 | sync selftest:`if len(real) >= 5` 静默跳过整个块,无 SKIP 消息;fresh ledger 上这部分 selftest 是 no-op |
| N11 | harness.py:42-43 vs main() | 口号 "a check that cannot run is a FAILURE, never a pass" 每次运行打印,但 SKIP 退出码是 0。崩溃=FAIL、环境不符=SKIP 的区分可以辩护,口号是假的 |
| N12 | harness.py:1769-1785, 2027-2039 | sft_pack_uncontaminated / template_contamination:asserts 说 "no holdout question appears verbatim",方法是 40 题抽样。docstring 诚实,asserts 夸大 |
| N13 | harness.py:3627-3641 | pod_drift 坏世界只覆盖 pod 分支;CI 分支(check_head)无坏世界,selftest 不覆盖 |
| N14 | harness.py:4298-4306 | device_set_honoured:`_CVD_SAFE` 不识别带引号的安全赋值(`CUDA_VISIBLE_DEVICES="${_DEVS[$i]}"` 组以 `"` 开头 → 误 FAIL);`"_DEVS" in l` 子串代理 |

## 框架机制评估(逐项)

- **CHECKS 注册表**:43 条五元组(name, asserts, incident, fn, broken),incident 字段是真实事故叙事——质量高。注册表与 STAGES 脱钩:STAGES(4806-4816)引用 check 名但无机制保证名字存在(agents_rules_covered 的 known 集合同理,白名单字符串无校验)。
- **--selftest 机制**:三层——坏世界必须 FAIL(6015-6043)、坏世界必须含 repo-real 路径的文件(6022-6028,防手写世界)、空转 PASS 元检查自带失败用例 fake_vacuous_pass(6063-6081)。骨架是好的。洞:空转元检查只覆盖 "数字 空格 字母" 格式(6054-6059 自认),无数字 PASS(curl_ipv4、no_stale_running、score_matrix 等)不可见——注释自己给出了正解(structured counts),未做。
- **坏世界构造**:普遍用**真工件变异**(真 train.py 改 ffn_hidden、真 exp.py logger 写行、真 git plumbing 造 merge),不是手写——这是本框架最强的部分。`_broken_merge_complete` 是教科书级的。
- **SKIP 语义**:崩溃→FAIL(4831-4832,fail-closed,好);超时→SKIP(D5,坏);环境不符→SKIP(可辩护,但口号撒谎,N11)。
- **cfg_default**:AST 读 train.py,缺字段 raise 不返回 None(645-657)——正确,且有专门 selftest(no_duplicate_defs 的坏世界)。洞:`@functools.lru_cache` 以 field 为键、ROOT 是全局——selftest 里 `ROOT = d`(5940)后缓存污染是潜在的;`experiments()`(675)同病。
- **规则覆盖表**:见 Headline 2。`_MANUAL_BASELINE = 22` 字面量棘轮(162)是对的形状;`_agents_rule_bullets` 的解析(172-200)依赖章节标题字面量,标题改名 → FAIL(loud,可接受)。
- **pre-commit hook**:staged-blob 跑 pod_drift(不跑工作区副本,防指纹规则反噬)、integration-tree 身份判定、180s 外层超时。洞:外层超时/异常退出时不命名 check(Headline 1 的第 2 层修复点);hook 成功时 harness 输出全弃,SKIP 不可见。

## 没活下来的指控(自查)

- task done 可自审:`cmd_task` 3884-3886 拒绝 reviewer==owner,CLI 层已强制。撤回。
- is_pod 身份"无 .git 即 pod"看似脆,但 selftest 是兜底(坏世界忘建 .git → SKIP → untested → 响亮失败),且 docstring 记录了 worktree 误判史。降级为 nit 不单列。

## 结论

骨架(A 级)和超时路径(F 级)的反差是本框架的现状:坏世界构造、棘轮、fold 规则都是同类最佳,但每个 check 头上的闹钟会把整个房子炸掉且不留字条,而规则覆盖表把三条规则挂在了不执行它们的 check 上。修 Headline 1 的三层 + Headline 2 的三行重映射之前,这个框架的"全绿"不具备它声称的含义。

## Follow-ups(2026-09-01,tilerl-0a 提案 + 修正)

1. **Call-shape-fidelity meta-check**(P6 扩展):现有 vacuous-PASS meta-check 只验"check 在它的 broken world 里会 fail",验不出失败世界的**调用形状**在生产里不存在。活标本:3b 1ef8cc1 的 flock 修复——selftest 写 `held_fd = _build_lock(...)` 绑定返回值(5/5 全绿),生产两个调用点(build_corpus.py:387、:639)全部丢弃,fd 被 CPython 立刻 close,flock 当场释放。tilerl 已拒。判据(tilerl 修正,取代本审的单侧 grep 提案):**∃ fn:(selftest 调用点全部绑定返回值)∧(生产调用点全部丢弃)**——信号在差集不在任一侧的绝对形状。AST 层面可算(`ast.Expr` 包 `ast.Call` = 丢弃,`ast.Assign` = 绑定),假阳性率低于单侧扫描;启发式标记人工裁决,不自动判死(绑定/丢弃的合法性依赖语义)。
2. **病因层(不立项,review 判断)**:返回值承载生命周期的函数应当是 context manager——`with build_lock(out):` 让"丢弃返回值"在语法层不可能犯。meta-check 抓的是症状,这条是病因。写进 lessons,不进 harness。
