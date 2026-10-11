# 100B Engram 预训练入口

## 结论

Stage-3 已稳定运行。当前运行名是 `s3a_seg1`。训练从 `ckpt_v42s2.pt` 的 step 54051 恢复。首个 10B token 评测门是 step 66767。

## 当前状态

| 项目 | 已验证值 |
| --- | --- |
| 训练目标 | 累计 100B tokens |
| 当前分段 | 从 step 54051 到 step 66767 |
| 最新验收点 | step 54080，累计 42.53B tokens |
| 模型参数 | 3,910.9M |
| Engram 层 | 1、5、9、13、17、21 |
| GPU | 8 卡，world=8 |
| 每步 tokens | 786,432 |
| 精度与算子 | FP8、FlashAttention |
| 首段数据 | 9 个许可 allowed 数据域 |
| 数据供给 | 57.52B tokens |
| 运行 claim | `s3a_seg1` 占用 0–7 卡 |
| 日志 | `/work/aupai/runs/s3a_seg1.log` |
| 退出码 | `/work/aupai/runs/s3a_seg1.rc`；运行中不存在 |

## 权威文件

| 用途 | 路径 |
| --- | --- |
| 数据 mix | `data/mix_s3a.json` |
| 训练脚本 | `runs/s3a_seg1.sh` |
| claim 监督器 | `runs/launch_s3a_claimed.py` |
| 旧 SFT 隔离脚本 | `runs/isolate_obsolete_sft.py` |
| 基础 checkpoint | `/work/aupai/ckpt_v42s2.pt` |
| 运行日志 | `/work/aupai/runs/s3a_seg1.log` |
| 宿主清理审计 | `/work/aupai/runs/obsolete_recovery_cleanup_20261011.txt` |

权威代码分支是 `stage3-100b`。权威工作区是 `/Users/bytedance/code/aupai_stage3_main`。

## 训练契约

1. 保留 Engram。不得删除或关闭 Engram 层。
2. 使用完整 100B 学习率计划。`max_steps` 固定为 127157。
3. 每新增 10B tokens 停止一次。首个停止点是 step 66767。
4. 每个停止点执行统一评测。评测完成前不得进入下一段。
5. 连续两次无提升时暂停训练。能力明显退化时立即暂停。
6. 只使用许可 allowed 的数据域。许可 unknown 数据保持隔离。
7. 旧 SFT、tiny CSA2 和远端 CI 不阻塞主线。
8. 不修改 MLX。
9. 合成数据默认使用 ASD-STE100。

## 验收命令

在 pod 中执行：

```bash
cd /work/aupai
python3 scripts/card_claim.py status
tail -120 runs/s3a_seg1.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
sha256sum runs/s3a_seg1.sh data/mix_s3a.json runs/launch_s3a_claimed.py
```

运行健康时必须同时满足：

- claim 名为 `s3a_seg1`，卡范围是 0–7。
- 父进程 `launch_s3a_claimed.py` 和子进程 `run_ddp.sh` 均存活。
- 日志 step 持续增加。
- 8 卡持续有训练负载。
- 日志无 NaN、OOM、NCCL 错误或未处理的 Traceback。
- `runs/s3a_seg1.rc` 不存在。

## 旧恢复链处理

旧 SFT 恢复链曾反复覆盖启动器并发送 SIGTERM。已移除以下宿主资产：

- 4 个隐藏 wrapper。
- 4 个 Python 守护进程。
- 4 个每分钟 cron。
- `stage_watchdog` 源目录。
- `.system-state` 缓存。
- 两个兼容服务已停止。

pod 内保留可逆副本。活动路径使用目录占位：

- `runs/ladder/guard/d1e_chain.sh`
- `sft_math.py`

恢复旧 SFT 前，必须先停止 100B 训练并释放 `s3a_seg1` claim。随后解除目录 immutable 属性，并从 `.disabled-for-100b` 副本恢复文件。不得在 100B 训练运行期间恢复这些入口。

## 下一步

1. 持续监控 step、loss、吞吐、显存和异常。
2. 在周期保存点确认 checkpoint 可恢复。
3. 在 step 66767 停止训练。
4. 执行统一评测。
5. 根据评测结果决定下一段 10B tokens。
