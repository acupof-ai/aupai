# 100B Engram 预训练入口

## 结论

当前活动运行为 `s3a_seg1`。训练保持 world8 到 step60000。step60000 checkpoint 稳定后，自动切换到 pod + H20B 的 world16。目标仍为 step66767。每个 checkpoint 自动执行 CPU 分层评测。

## 当前状态

| 项目 | 当前值 |
| --- | --- |
| 更新时间 | 2026-10-12 19:28 CST |
| 活动运行名 | `s3a_seg1` |
| 当前 step | 57340 |
| 停止 step | 66767 |
| 剩余 step | 9427 |
| 最新 checkpoint | `ckpt_s3a_seg1.pt.step56000` |
| 无损切换 checkpoint | step60000 |
| checkpoint 间隔 | 4000 step，按绝对 step 对齐 |
| world8 速度 | 约 10.3 s/step |
| 预计到达 step60000 | 2026-10-13 03:05 CST |
| world16 预估速度 | 约 5.15 s/step，切换后实测 |
| world16 预估停止点 | 2026-10-13 13:00 CST，含切换余量 |
| GPU | 当前 8 卡 world8；切换后 16 卡 world16 |
| 每 step tokens | 786,432 |
| Engram 层 | 1、5、9、13、17、21 |
| 数据 mix | `data/mix_s3a.json` |

## 本地与运行期映射

| 用途 | 本地权威路径 | 运行期路径 |
| --- | --- | --- |
| 训练脚本 | `runs/c100_seg1.sh` | `runs/s3a_seg1.sh` |
| claim 监督器 | `runs/launch_c100_claimed.py` | `runs/launch_s3a_claimed.py` |
| 数据 mix | `data/mix_c100.json` | `data/mix_s3a.json` |
| 工作目录 | `/Users/bytedance/code/aupai_stage3_main` | `/work/aupai` |

本地脚本保留段启动配置。运行期脚本可指向最新 checkpoint。两者必须保留 Engram 参数。

## world16 自动切换

| 项目 | 内容 |
| --- | --- |
| node0 | pod 节点 `192.168.29.37`，ranks 0-7 |
| node1 | H20B 节点 `192.168.29.36`，ranks 8-15 |
| 切换点 | `ckpt_s3a_seg1.pt.step60000` |
| 等待器 | `runs/start_s3a_w16_switch_waiter.sh` |
| 切换脚本 | `runs/s3a_w16_switch.sh` |
| node0 claim | `runs/launch_s3a_w16_claimed.py` |
| node1 host 脚本 | H20B `/data00/aupai_work/aupai/s3a_w16_node1.sh` |
| 精度 | 默认 FP8；保持当前训练轨迹 |
| 口径 | batch 4、accum 3、world16、seq 4096，每 step 786,432 tokens |

切换前必须满足三项条件：checkpoint 文件大小连续 10 秒不变；H20B corpus 校验通过；node1 容器先启动。切换后必须核验 16 卡显存、利用率、首个 step 和日志错误。

## 自动评测

| 层级 | 检查内容 | 触发点 |
| --- | --- | --- |
| 配置检查 | checkpoint step、`arch=v42`、Engram 层列表 | 每个 checkpoint |
| 权重检查 | 参数有限性、最大绝对值 | 每个 checkpoint |
| 分层状态检查 | 24 个 layer、6 个 Engram 的 persistent state | 每个 checkpoint |
| 分层激活检查 | 固定短文本前向，激活有限性、均值、最大值 | 每个 checkpoint |
| 固定文本 NLL | 4 条 ASD-STE100 风格技术句 | 每个 checkpoint |
| 多领域 NLL | code、math、technical、Chinese、dialogue、prose | 每个 checkpoint |
| 普通对话生成 | 2 条固定问题，greedy 生成 32 token | 每个 checkpoint |
| 输出长度与重复度 | 长度、EOS、unique token、连续重复、4-gram、重复行 | 每个 checkpoint |
| 固定多选快测 | 8 题，覆盖常识、科学、数学、代码、中文、流程、逻辑、安全 | 每个 checkpoint |
| 标准多选 quick32 | HellaSwag、PIQA、ARC-Easy、ARC-Challenge、WinoGrande 各 32 题 | 每个 checkpoint |
| access-code canary | 固定多选记忆信号 | 每个 checkpoint |
| HumanEval quick8 | 前 8 题 greedy pass@1 | 每个 checkpoint |
| HumanEval full164 | 164 题 greedy pass@1，输出 dc156 和 all164，写入 `eval_watch.jsonl` | 每个 checkpoint，quick32 后顺序执行 |

关键文件：

| 用途 | 路径 |
| --- | --- |
| CPU smoke | `scripts/cpu_ckpt_smoke.py` |
| quick benchmark 构建器 | `scripts/build_quick_bench.py` |
| 离线 quick 数据读取 | `eval/local_bench.py` |
| 分层评测 driver | `scripts/tiered_ckpt_eval.py` |
| watcher 启动器 | `runs/start_tiered_ckpt_eval.sh` |
| HumanEval 分片合并器 | `runs/merge_humaneval_shards.py` |
| 正式 HumanEval ledger | `/work/aupai/runs/eval_watch.jsonl` |
| tiered ledger | `/work/aupai/runs/tiered_ckpt_eval_s3a_seg1.jsonl` |

## step55000 已验证结果

| 评测 | 结果 |
| --- | --- |
| CPU smoke | 通过 |
| finite parameter fraction | 1.0 |
| 固定文本 mean NLL | 3.305436 |
| 多领域 mean NLL | 2.988640 |
| 普通对话 | 2 条均生成 32 token；warning：未到 EOS |
| 输出重复度 | unique token fraction 0.56、0.59；无长重复 |
| access-code canary | 正确，margin 2.681441 |
| HumanEval quick8 | 4/8 = 50.0%，空输出 0/8 |
| HumanEval 全量 | 43/164 = 26.22%，空输出 0/164 |

正式 gate 使用 156 题去污染口径。当前 43/164 为全量口径。汇报正式 gate 前，必须读取 `runs/contam_r3_he_union.json` 并重新过滤。

## step56000 已验证结果

| 评测 | 结果 |
| --- | --- |
| CPU smoke | 通过 |
| 固定多选快测 | 3/8 = 37.5% |
| HumanEval quick8 | 4/8 = 50.0%，空输出 0/8 |
| HellaSwag quick32 | 31.2% |
| PIQA quick32 | 68.8% |
| ARC-Easy quick32 | 34.4% |
| ARC-Challenge quick32 | 9.4% |
| WinoGrande quick32 | 56.2% |
| 五项平均值 | 40.0% |
| HumanEval dc156 | 44/156 = 28.21% |
| HumanEval all164 | 44/164 = 26.83%，空输出 0/164 |

quick32 只用于趋势监控。样本量小，结果不能替代完整 benchmark。普通对话仍全部达到 32 token 上限，状态为 warn。

## step66767 统一评测

训练停止后，按顺序执行下列命令。所有结果写入 `runs/` 和 `data/eval/`。评测完成前，不启动下一段。

| 顺序 | 评测 | 命令 |
| --- | --- | --- |
| 1 | CPU smoke | `python3 scripts/cpu_ckpt_smoke.py --name s3a_seg1 --once --threads 4` |
| 2 | HumanEval full164 | `python3 scripts/eval_watch.py --name s3a_seg1 --device cpu --step 66767 --once --workers 4 --threads 40 --timeout 7200` |
| 3 | 标准 GPU 套件 | `python3 eval/run_eval.py --ckpt ckpt_s3a_seg1.pt.step66767 --device cuda --batch 32 --benchmarks hellaswag piqa arc-easy arc-challenge winogrande boolq openbookqa mmlu gsm8k --json_out runs/s3a_seg1_step66767_standard.json` |
| 4 | 分领域 loss | `python3 eval/domain_loss.py --ckpt ckpt_s3a_seg1.pt.step66767 --mix data/mix_s3a.json --json runs/s3a_seg1_step66767_domain.json` |

正式代码门只采用 `eval_watch.jsonl` 的 dc156 结果。标准 GPU 套件用于能力画像，不替代代码门。

## 训练契约

1. 必须保留 Engram 层 1、5、9、13、17、21。
2. `max_steps` 固定为 127157。
3. 到 step66767 必须停止。
4. 停止后执行统一评测。评测完成前不得进入下一段。
5. 连续两次无提升时暂停训练。能力明显退化时立即暂停。
6. 只使用许可 allowed 的数据域。
7. 旧 SFT、tiny CSA2 和远端 CI 不阻塞主线。
8. 不修改 MLX。
9. 合成数据默认遵循 ASD-STE100。

## 运维限制

- 对整个训练进程组发送 SIGTERM 未生成 interrupt checkpoint。该次运行从 step57730 回退到 step55000。
- 后续停止操作必须先确认保存路径和保存逻辑。不得假设 SIGTERM 会生成 interrupt checkpoint。
- compressor 的 non-persistent `score_state` 使用 `-inf` 哨兵。该值不是训练故障。
- HyperConn 参数保持 fp32。不得将 CPU checkpoint 的全部浮点参数强制转为 bf16。

## 巡检命令

```bash
cd /work/aupai
python3 scripts/card_claim.py status
tail -n 80 runs/s3a_seg1.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls -l runs/s3a_seg1.rc 2>/dev/null || true
tail -n 20 runs/tiered_ckpt_eval_s3a_seg1.jsonl
```

健康条件：step 增加；claim 持有 0–7 卡；8 卡有负载；rc 不存在；日志无 NaN、OOM、NCCL 错误、未处理 Traceback；watcher 存活。
