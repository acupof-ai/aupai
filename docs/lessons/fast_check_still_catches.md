---
question: 把一个慢检查改快之后,怎么知道它还抓得住原来那个缺陷——四次"看起来对"的快版本,三次在缺陷上 PASS
status: measured
source: tilerl 2026-09-01,spawned_scripts_exist 7650ms→13ms(8548ac8)。原缺陷 c3a47e8
---

# 加速一个检查,等于重写它;重写之后它防的东西默认为零

## 起因

`spawned_scripts_exist` 连续 8 次在 5s 预算上超时,报的是"has not actually run
since"——一个**没有任何修复能清掉的红**,同时它挡住全仓所有提交。

慢因是一个 import,不是解释器启动:

| | 秒 |
|---|---|
| `fla.ops.kda`(经 `import train` 传递进来) | 6.07 |
| torch 本身 | 0.92 |
| 全部六个脚本串行 | 7.65 |
| 三个子进程开线程 | **11.4**(更慢:争用,不重叠) |

抬 deadline 会把一个坏掉的检查伪装成一个慢检查。

## 三个快版本在它专防的缺陷上 PASS

缺陷树(c3a47e8 那棵):`harness.py` 在 `scripts/`,脚本在 `datagen/`,脚本只
`sys.path.insert(0, ROOT)`——所以它 `import harness` 必失败。

| 版本 | 结果 | 为什么 |
|---|---|---|
| `find_spec` + 前置 sys.path | **PASS** | find_spec 读的是**调用进程**的 sys.path,harness.py 自己的 path 上有真的 scripts/ |
| `find_spec` + 替换 sys.path | **PASS** | find_spec 先查 `sys.modules`,harness.py 早就 import 过这些模块了 |
| `PathFinder` + 按搜索路径判归属 | **PASS** | harness.py 在 scripts/,坏脚本恰恰不 insert 它 → 被判为第三方 → 跳过 |

第三个最隐蔽:归属判据本身把唯一能暴露缺陷的那个名字排除掉了。**筛掉第三方**和
**筛掉脚本够不着的东西**看起来是同一件事,实际是相反的——够不着正是要测的属性,
不是过滤条件。

判据改成「repo 里任何位置有这个文件 = 我们的」,第四版才红。

## 规则

1. **改快一个检查,必须重建它原来抓住的那棵树并断言 FAIL。** 断言真树 PASS 是不够
   的——三个错版本全部满足"真树 PASS 且很快"。
2. **`find_spec` 不能用来问"另一个进程能不能找到这个模块"**:它读调用者的 path,
   且先查 import 缓存。要显式传搜索路径的 `PathFinder.find_spec(name, paths)`。
3. **快检查要写明自己覆盖什么。** 这版抓「找不到的导入」,抓不到「能导入但执行时
   抛」。后者要花那 9s。ceiling 写进代码,不藏。
4. **缓存慢检查 ≠ 修好慢检查。** 被换掉的 298409c 用六个文件的字节做 key 缓存 9s
   的结果,绿是同样的绿,但盲区是真的:这六个文件**导入的**东西坏了,要等六个文件
   之一变动才看得见。13ms 之下缓存没有东西可省,盲区却留着。
5. 同一条的推论:**一个没人跑得起的检查,不如一个说清自己边界的快检查**——它挡了
   8 次提交,期间没有防住任何东西。

## 连带

`test_spawned_fast.py` 重建 c3a47e8 树断言 FAIL,并断言真树 `< 2.0s`(不是 `< 5s`:
卡在预算边上等于下一个 import 就再次超时)。第四个"看起来对"的版本落不进来。
