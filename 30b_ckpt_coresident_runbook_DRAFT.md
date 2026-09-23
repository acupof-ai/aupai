# 30B CED run — checkpoint 磁盘管理 & co-residency runbook（DRAFT，0e，未进 git，待 08 审）

> 状态：只读规划，**一个 checkpoint 都不删**。所有数字读自 `origin/main` 代码与 pod 2026-09-22 实测。
> 适用 run：CED 重训（`runs/v41_gate_0922.sh` 形态，CED 落地后换模型不改这些机制），8×H20，38146 步，`--save_every 2000`，单整存 12.97 GB（实测 flat ckpt）。

---

## 0. 实测基线（2026-09-22，写方案前）

- `/work` 与 `/data00` 是**同一个 device** `/dev/nvme0n1`（3.5T，used 1.7T，**avail 1.7T = 50%**）。不是两块盘。
- `/mnt/data02` **不存在**：`/mnt` 为空目录，`mount` 无 data02，`df /mnt/data02` 失败。**第二块 NVMe 当前没挂**（§3 的备份前提不成立，先挂盘）。
- 现存 flat ckpt 49 GB / 4 个（step4000/6000/8000 各 12,973,995,063 B，interrupt.step8196 12,974,000,917 B），**一个不删**。
- 六 gate cache 在 `/data00/tokens_<domain>.pt`（**注意：直接在 /data00，不在 token_cache 子目录**），四戳全对（vocab f1f860970d15d623 / seed 42 / srcfp==live fp_dir），总 86.46B int32 token。
- 卡 7 被 tilerl-etx 线占 325 MiB（活 claim `runs/claims/tilerl-etx-reserve.7.json`），卡 0-6 空。S0/single-card smoke 用 0-6 不受影响；world-8 需用户协调释放。

---

## 1. checkpoint 磁盘管理（答案：滚动删除已在代码里，不要新造）

### 1.1 仓里已有的事实（`train.py`，读代码不是建议）

- **周期性整存**：`train.py:4172` 每 `--save_every`（2000）步在 rank0 写 `ckpt_path + f".step{step}"`，含 model+opt+cursor。
- **保留最近 3 个，自动滚删**：`train.py:4174-4199` 内建 roller——
  - `stale = sorted(glob(ckpt_path + ".step*"))[:-3]`，对超出最近 3 个的 `.step*` 直接 `os.remove`。
  - **milestone pin 豁免**：删前把候选 inode 与目录下所有 `*.milestone_*.pt` 硬链的 inode 比对，命中则保留（log `roller: keeping ... -- pinned`）。
- **interrupt 存盘不参与滚动**：SIGTERM/SIGINT handler（`:3915-3927`）写 `.interrupt.step{N}`，roller 的 glob 只匹配 `.step*`。**实测：`.interrupt.step8196` 与 step8000 并存未被删，证实此点。**
- **epoch / final 存盘不参与滚动**：每 epoch 写 `.ep{N}`（`:4531`），run 末写裸 `ckpt_path`（final，`:4547`）；roller 只 glob `.step*`，这两类永不被它删。
- **写盘非原子**：`save_checkpoint` 直接 `torch.save(ck, path)`（`:1948`），**无 .tmp+rename**。写一半被杀会留截断文件。

### 1.2 30B 的占用推演（save_every 2000 × 38146 步）

| 类别 | 数量 | 单 12.97GB | 说明 |
|---|---|---|---|
| 滚动 `.stepN` | 恒 **3** | ~38.9 GB | roller 自动保持最近 3，不再增长 |
| `.interrupt.step*` | **不自动清** | 每次中断 +12.97 | 多次 kill/restart 会**累积**，roller 不碰 |
| `.epN` | 每 epoch 1 | ×epochs | 不被 roller 删 |
| final 裸 `.pt` | 1 | 12.97 | run 末写 |
| milestone 硬链 | 硬链占 0 额外字节 | — | pin 用同 inode；只有真复制才占空间 |

**结论：稳态滚动只占 ~39 GB**（3 个 step）+ interrupt/ep/final。1.7T 余量**无缺口**，不需要靠手删滚动整存腾地方——代码已经在做。

**真正的增长风险是 `.interrupt.step*`**：30B 若被反复 stop/resume，每次留 12.97 GB 且无人清。~130 次中断才会逼近 1.7T，但这是唯一需要人管的累积项。

> **运维规则（08 定 2026-09-22）：每次成功 resume 之后**，旧的 `.interrupt.step*` 件才进入删除评估——先经 `scripts/deletion_candidates.py` 五谓词，再经用户/owner 点名，二者都过才删；**不自动 rm**。当前未 resume 的 interrupt 件保留作 resume 源。
>
> **save 策略已定（08）**：CED 沿用 `--save_every 2000`（门线不改）；命名走 `ckpt_{name}.pt` 滚动 + milestone 硬链（即 §1.1 现有机制）。

### 1.3 谁在何时删 / 怎么不删到还在被引用的 ckpt

**不要手写 `rm ckpt_*.step*`。** 仓里有专门的安全删除判定器 `scripts/deletion_candidates.py`，它把一个 ckpt 标为 protected 的五条来源（`protections()`）：

1. **PINNED** —— 出现在 `runs/milestones.jsonl`（ckpt / pinned_as 字段）；
2. **CITED by any ledger** —— 被任何 `runs/*.jsonl` 引用（结果引用的 ckpt 必须可加载）；
3. **RESUME SOURCE** —— 被某记录 run 的 `--resume PATH`（同时认 `--resume=PATH` 两形式）引用，**已结束的 run 也算**（trajectory 仅在源存在时可重建）；
4. **NAMED BY OPEN ROW** —— 被 status=running 的 experiments 行点名（可能正在写/读）；
5. **HARDLINKED** —— `st_nlink>1`（milestone 硬链等）。

**规定（fact/源规则，照已有工具）：**
- 滚动删除交给 train.py roller，**人不碰 `.step*`**。
- 只在 interrupt 累积需要回收时，先跑 `python3 scripts/deletion_candidates.py` 拿到带 `protected_by`/`family_cited` 的清单，**只对 protected_by 为空、family_cited 为 None、且不是当前 resume 源的条目**，经**用户点名**后删；删前 `stat -c '%s %n'` 写 `deleted_*` manifest（本仓既有规矩）。
- 当前 resume 链：run 从 `--resume ckpt...step8000`（或 CED 重启指定的那个）起，**那个源在整条 trajectory 结束前永远 protected（谓词 3），绝不可删**。
- 删任何 ckpt 前确认没有 status=running 的行点名它（谓词 4）。

### 1.4 milestone / final 备份到第二盘

工具已存在：`scripts/pod_backup.sh`（在 pod 上 `bash scripts/pod_backup.sh`），它已处理：
- `require_durable` + `is_mounted`：`$AUPAI_BACKUP_DEST`（默认 `/mnt/data02`）**必须是真挂载点**，否则拒绝（防止备到同 emptyDir，2026-08-30 已付过一次学费）；
- `backup_paths` 只备 **final 裸 `.pt` + milestone**（`ls ckpt_*.pt | grep -v '\.pt\.'`，自动排除 step/interrupt/ep），外加 runs/*.jsonl、facts、tokenizer、mix；
- `_settle`：size 3s 稳定才拷，**拒绝还在长大的 ckpt**（防把训练末端正写的 final 拷成截断件）；
- 写 MANIFEST（含 sha256），mtime 被 harness 的 check_root_durable 盯。

**时机规定：**
1. **第二块盘挂载属基础设施，本次训练前不处理（08 定）**：`/mnt/data02` 自 pod 0916 重建后就没了，不在 CED 训练前挂。**final/milestone 备份因此【前置未满足、暂不启用】**，等盘挂回（`mountpoint -q /mnt/data02` 为真）再恢复本节；在此之前 final/milestone 只活在单盘 /data00（emptyDir/hostPath 风险知悉）。pod_backup.sh 在盘未挂时会且应该拒绝，不要绕过。
2. **milestone**：每个要长期保留的点（如 30B 关键 token 刻度）先用 harness 的 milestone 机制把 `.stepN` **硬链**为 `ckpt_<run>.milestone_<token>.pt`（roller 即据此 pin，同 inode 不额外占 /data00），然后跑 pod_backup 复制到 data02。
3. **final**：run 末裸 final 写完且 `_settle` 确认不增长后，跑 pod_backup。
4. 不要在训练写盘窗口里跑备份（_settle 会等，但避开更稳）。

---

## 2. smoke / 训练期 co-residency 守护 runbook

### 2.1 已有的门（`eval/cache_guard.py`，不要新造阈值）

- **新鲜度门 `assert_caches_fresh`**（`scripts/loader.py`）：每个 `tokens_<d>.pt` 三件套 `.vocab`/`.srcfp`/`.seed` 必须匹配——vocab == gate f1f860970d15d623、`.srcfp` == 对域 live fp_dir、`.seed` 一致；train 启动前必过。**当前六域全过（K8 已核）。**
- **co-residency 门 `assert_not_co_resident`**：当某 eval 要读的 cache 字节 `>= CO_RESIDENCY_BYTES = 10e9`（**10 GB**，`scripts/eval_load_cost.py:127`）且有**活 claim 占卡**时，**拒绝**（`CoResidentCacheRead`）。
  - 旁路：`AUPAI_ALLOW_CORESIDENT_CACHE=1`（仅当 controller 明确借卡），不要自己设。
  - 判活读 `card_claim.py` 的 claims（死 pid 算 stale，zombie 不算）。

### 2.2 关键：按"实际读入字节"判，不是文件大小

`eval_load_cost.head_read_bytes`：cache 用 `torch.load(mmap=True)`，驻留成本随**取的行数**变，不随磁盘文件大小。
- **val-head 读**最多 `SEQ_CAP=64` 序列（`eval/domain_loss.py:57`）≈ **2 MB**（64×4097×8 B），**无论该域磁盘上多大** → 永远 < 10 GB，**不触发** co-residency 门。
- 全量文件读（ppl 全库等）按文件大小计价 → 大域（l2 144.6GB、l3 106.8GB 等）远超 10 GB，**触发**。

### 2.3 共存白/黑名单（smoke 与 30B 训练同机时）

| eval | cache 读 | 与训练共存？ |
|---|---|---|
| `eval/domain_loss.py`（val-head 64 序列） | ~2 MB/域 | **可共存**（不触发门；只读 ckpt+小头） |
| `eval/api_cloze.py` 等 head 型 | 小头 | 可共存（先 `scripts/eval_load_cost.py` 确认 want<10GB） |
| `eval/ppl.py`、`eval/domain_bpb.py` 全量 | 全文件（l2/l3 上百 GB） | **不可共存**：co-residency 会拒，或需 controller 借空闲卡 + `AUPAI_ALLOW_CORESIDENT_CACHE=1` |
| `eval/score_matrix.py`、`nan_probe.py` | 看具体 domains | 起前用 eval_load_cost 估价 |
| `scripts/pretokenize_domains.py` | 写 cache（CPU/IO 重） | **不要与训练叠**：吃满核 + 写 /data00 同盘（device 66307/nvme0n1，与训练 ckpt 写争 IO） |

**硬规矩：**
1. 任何 eval 起前先 `python3 scripts/eval_load_cost.py <domains/head_rows>` 看 want GB；**want ≥ 10 GB 且训练有活 claim 就不要起**，让门拒或找 controller 借卡。
2. **读大 token cache 的活等训练**：l2/l3 这种上百 GB 的全量 cache 读，不在训练跑时做（IO 与驻留双争），排到训练停或专用卡窗口。
3. head 型 eval 可在空闲卡（当前 0-6）上跑，但仍走自己的 claim，不要假设卡空闲就能无 claim 读。
4. cache_guard 的阈值（10 GB / 64 序列）从代码读，不手抄、不在 runbook 里改；要改走 PR。

---

## 3. 已定 & 待确认

**已定（08 2026-09-22）：** CED 沿用 save_every 2000 + `ckpt_{name}.pt` 滚动/milestone 硬链；/mnt/data02 本次训练前不挂、final/milestone 备份前置未满足暂不启用；.interrupt.step 每次成功 resume 后经 deletion_candidates 五谓词+点名才删（不自动 rm）。

**待确认：**
- [ ] milestone 的 token 刻度表（哪些点承诺长期保留）——属配方/检查点策略，由用户定（08 world8 前问）。
- [ ] world-8 时卡 7 tilerl-etx 占用的释放协调（08 回用户，不自行 kill）；66 可能需要 host 视图配合读 nvidia-smi UUID。

---

## 4. 一句话给值班

滚动整存不用管（代码保留最近 3 + milestone pin）；要删只用 `deletion_candidates.py` 且用户点名；要备份先挂 data02 再 `pod_backup.sh`（只备 final/milestone、自带防截断）；eval 共存看 `eval_load_cost` 的实际读入字节，head(~2MB) 可共存、全量 cache(≥10GB) 等训练；六 cache 新鲜度门当前全过。
