# awb 与 tmux 常用操作手册

本手册记录监工的常用命令。目标：快速定位 worker，读取状态，推进任务。

## 1. 环境变量

每个 awb 命令前先设置两个变量。

```bash
export AWB_DIR=/Users/bytedance/.aupai-team/awb
AWB=/Users/bytedance/.local/bin/awb
```

Pod 命令统一走 `~/bin/pod`。容器内代码路径是 `/work/aupai`。Pod 命令只用 ASCII。

## 2. awb 常用命令

| 目的 | 命令 |
| --- | --- |
| 查看所有 worker | `$AWB peers` |
| 派一个任务 | `$AWB send <id> "<任务一行描述>"` |
| 直接投递消息 | `$AWB tell <id> "<消息>"` |
| 闭环并报结果 | `$AWB reply <id> <key> "<结果一行>"` |
| 规划任务状态 | `$AWB task <id> todo "<步骤>"` |
| 放弃任务 | `$AWB task <key> drop "<原因>"` |
| 标记阻塞 | `$AWB block <id> "<原因>"` |
| 解除阻塞 | `$AWB unblock <id>` |
| 验收 | `$AWB check <id> -- <命令>` |

规则：

- 一个 worker 同时只开一个任务。
- worker 回复必须带 key，格式是 `[awb <id>#<key>]`。
- 带 key 的进度报告不是结果。任务保持打开。
- 只有 reply 能关闭任务。

## 3. tmux 常用命令

awb 的主 session 名是 `aupai`。

| 目的 | 命令 |
| --- | --- |
| 列出 session | `tmux ls` |
| 列出 pane | `tmux list-panes -t aupai -a -F '#{window_index}.#{pane_index} #{pane_title}'` |
| 抓取 pane 内容 | `tmux capture-pane -t aupai:<w.p> -p` |
| 按回车 | `tmux send-keys -t aupai:<w.p> Enter` |
| 输入文字 | `tmux send-keys -t aupai:<w.p> "<文字>"` |
| 移动光标 | `tmux send-keys -t aupai:<w.p> Down` |
| 接入 | `tmux attach -t aupai` |

也可用助手脚本 `scripts/tmux_ops.sh`。

```bash
bash scripts/tmux_ops.sh panes
bash scripts/tmux_ops.sh capture 2.3
bash scripts/tmux_ops.sh enter 2.3
bash scripts/tmux_ops.sh choose 2.3 2
```

## 4. worker 卡住的处理

worker 常见的两种等待状态。

### 4.1 选择菜单

worker 弹出选项，等待选择。屏幕底部显示 `Enter to select`。

处理步骤：

1. 用 `capture` 读取选项。
2. 选定后用 `enter` 确认。
3. 出现汇总页时，选 `Submit answers`。
4. 底部出现 `bypass permissions on` 后，后续不再弹框。

### 4.2 权限对话框

`peers` 显示 waiting。awb 提示 `waiting on a permission dialog`。

处理：用 `enter` 应答，或在 pane 内手动允许。权限绕过后，worker 自动继续。

## 5. 开 PR 与合并

改动在独立 worktree 完成。流程如下：

1. 建 worktree：`git worktree add <路径> -b <分支> e180a372`。
2. 在 worktree 内改动并提交。
3. 推送：`git push -u origin <分支>`。
4. 开 PR：`gh pr create`。
5. 验收通过后合并：`$AWB merge <PR> --watch`。

规则：

- 不使用 force push。
- 一个任务一个分支。
- 合并前确认验收命令退出码为 0。

## 6. 安全约束

- 不写入凭证。
- 不改训练与节点。
- 不碰他人容器 `linyu-dev`。
- HF 导出搁置，未收到通知不推进。
- H20B 临时数据用完即删。
