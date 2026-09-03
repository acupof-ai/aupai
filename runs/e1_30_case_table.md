# e1-30 case table: 12 held-out items with both arms' per-item loss

Population 10,421 items scored by both arms. Selection fixed in `scripts/e1_30_case_table.py`, seed 20260903. **nll per supervised byte**, each arm over its own formatting.

`ratio` = ctrl / ours: above 1 means WE did better on that item. "Our advantage smallest" is not "the control wins" -- read the `winner` column.

## our advantage largest (>= 100 B)

| id | class | ours | ctrl | ratio | winner | bytes | tokens (ours/ctrl) |
|---|---|---|---|---|---|---|---|
| 330350 | zh-code | 0.0471 | 0.8672 | 18.412 | ours | 898 | 318 / 299 |
| 479200 | zh-code | 0.0697 | 1.1480 | 16.479 | ours | 424 | 133 / 132 |
| 305600 | zh-code | 0.0853 | 1.3447 | 15.772 | ours | 240 | 85 / 90 |
| 377450 | zh-prose | 0.1034 | 1.4022 | 13.566 | ours | 2,287 | 582 / 1004 |

**330350** (zh-code, ratio 18.412, winner ours, 898 B)

> Q: 请根据描述和条件，给结下来输入的编程题提供解法 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。 示例 1： 输入：nums = [-10,-3,0,5,9] 输出：[0,-3,9,-10,null,5] 解释：[0...

> A: 以下是可供参考的实现方案： ['```java\n/**\n * Definition for a binary tree node.\n * public class TreeNode {\n * int val;\n * TreeNode left;\n * TreeNode right;\n * TreeNode() {}\n * TreeNode(i...

**479200** (zh-code, ratio 16.479, winner ours, 424 B)

> Q: 请根据描述和条件，给结下来输入的编程题提供解法 给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。 请你返回该链表所表示数字的 十进制值 。 示例 1： 输入：head = [1,0,1] 输出：5 解释：二进制数 (101) 转化为十进制数 (5) 示例 2： 输入：head = [0] 输...

> A: 以下是可供参考的实现方案： ['```python\n# Definition for singly-linked list.\n# class ListNode:\n# def __init__(self, val=0, next=None):\n# self.val = val\n# self.next = next\nclass Solution:\n...

**305600** (zh-code, ratio 15.772, winner ours, 240 B)

> Q: 请根据描述的需求和实现思路提示，完成函数开发 泰波那契序列 Tn 定义如下： T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2 给你整数 n，请返回第 n 个泰波那契数 Tn 的值。 示例 1： 输入：n = 4 输出：4 解释： T_3 = 0 + 1 + 1 = 2 T_4 = ...

> A: 以下是可供参考的实现方案： ['```python\nclass Solution:\n def tribonacci(self, n: int) -> int:\n a, b, c = 0, 1, 1\n for _ in range(n):\n a, b, c = b, c, a + b + c\n return a\n```']

**377450** (zh-prose, ratio 13.566, winner ours, 2,287 B)

> Q: 请分析给定的代码是什么编程语言, 函数实现的功能是什么？是用什么样的思路编写的？ 输入代码： ['```java\nclass Solution {\n private long[] p;\n private long[] h;\n\n public String longestPrefix(String s) {\n int base = 131;\n i...

> A: 这段代码使用了 Java 语言。 实现的时候主要使用了字符串哈希的想法。 这里提供一个参考的实现思路，**字符串哈希**是把一个任意长度的字符串映射成一个非负整数，并且其冲突的概率几乎为 0。字符串哈希用于计算字符串哈希值，快速判断两个字符串是否相等。 取一固定值 BASE，把字符串看作是 BASE 进制数，并分配一个大于 0 的数值，代表每种字符。一般来说...

## our advantage smallest (>= 100 B)

| id | class | ours | ctrl | ratio | winner | bytes | tokens (ours/ctrl) |
|---|---|---|---|---|---|---|---|
| 205100 | zh-prose | 2.0457 | 1.8845 | 0.921 | control | 222 | 121 / 93 |
| 343350 | en-prose | 1.1574 | 1.1592 | 1.001 | ours | 2,019 | 657 / 523 |
| 438700 | en-prose | 1.0160 | 1.0422 | 1.026 | ours | 2,505 | 774 / 595 |
| 374350 | en-code | 0.4147 | 0.4362 | 1.052 | ours | 2,517 | 827 / 703 |

**205100** (zh-prose, ratio 0.921, winner control, 222 B)

> Q: 所提供的文本是英文的，我们要求你将文本翻译成克罗地亚语。在翻译时请记住以下准则。1）我们想要一个自然的翻译，一个正式的形式。2）按原样使用'#@%$-+_=^&！*'等符号。在翻译成克罗地亚语时，*包括*适合的特殊字符。3) 像百万或十亿这样的数量应该翻译成克罗地亚语中的相应数量 4)请注意，除了特殊的占位符之外，输入都是区分大小写的，而输出也应该是区分大小...

> A: ['To nije ništa drugo nego nešto što ste stavili u jutarnjim satima, i to će vam dati dodatnu snagu, i to će dodatno pojačati vašu brzinu, i to će vam pomoći, na primjer, upravljat...

**343350** (en-prose, ratio 1.001, winner ours, 2,019 B)

> Q: 生成一首节日主题的诗歌。

> A: 'Twas the night before Christmas, and all through the house, Not a creature was stirring, not even a mouse. The stockings were hung by the chimney with care, In hopes that Saint Ni...

**438700** (en-prose, ratio 1.026, winner ours, 2,505 B)

> Q: As a technical writer helping draft educational content, for a technical documentation, I'm dealing with trying to understand ADHD medication options for a family member. I'm a tec...

> A: Happy to give a conceptual overview. This is general education, not medical advice — I'll flag at the end what genuinely requires a clinician. ## The two main classes **Stimulants*...

**374350** (en-code, ratio 1.052, winner ours, 2,517 B)

> Q: Given the code below how would you add a logo to the navbar and redirect or close the burger menu after an item is clicked?

> A: Here is an example of how you can add a logo to the navbar and redirect or close the burger menu after an item is clicked: ``` <nav class="navbar navbar-expand-lg navbar-light bg-l...

## uniform random, no length floor (seed 20260903)

| id | class | ours | ctrl | ratio | winner | bytes | tokens (ours/ctrl) |
|---|---|---|---|---|---|---|---|
| 130200 | en-prose | 0.2279 | 0.6746 | 2.960 | ours | 2,474 | 996 / 997 |
| 195050 | zh-prose | 0.6374 | 2.0208 | 3.170 | ours | 21 | 5 / 9 |
| 225800 | zh-prose | 1.5476 | 2.5485 | 1.647 | ours | 19 | 6 / 6 |
| 389150 | zh-prose | 0.4344 | 1.1666 | 2.686 | ours | 1,347 | 371 / 593 |

**130200** (en-prose, ratio 2.960, winner ours, 2,474 B)

> Q: For integers greater than 1, the sequence $a_n$ is defined as $\frac{1}{\log_n 2002}$. Let $b$ be the sum of $a_2$, $a_3$, $a_4$, and $a_5$, and let $c$ be the sum of $a_{10}$, $a_...

> A: <Thought> Alright, I need to solve the problem where for integers greater than 1, the sequence \( a_n \) is defined as \( \frac{1}{\log_n 2002} \). I have to find \( b - c \), wher...

**195050** (zh-prose, ratio 3.170, winner ours, 21 B)

> Q: 在这项任务中，你将得到一份制作一道菜所需的原料清单。你需要输出两样东西--菜肴的名称和可供多少人食用。如果清单中没有足够的细节来确定所做的是什么菜或可供多少人食用，则输出 "未知"。这道菜应该是煎饼、蛋糕或饼干中的一种。份数应该是2，8，或12。 成分列表。['4杯面粉', '1/2茶匙小苏打', '1杯糖', '3个鸡蛋'] 。

> A: 煎饼，12

**225800** (zh-prose, ratio 1.647, winner ours, 19 B)

> Q: 给你一个整数数组，你的任务是找到一对相邻元素的最大积。输出结果应该是21。 [3, 6, -2, -5, 7, 3].

> A: 3 * 6 = 1

**389150** (zh-prose, ratio 2.686, winner ours, 1,347 B)

> Q: 小明有10块钱，他想买一些橙子和一些苹果。橙子每个卖2块，苹果每个卖1块。他最多可以买几个橙子和几个苹果？

> A: 首先，小明有10块钱，橙子每个2块，苹果每个1块。 设小明买x个橙子和y个苹果。 根据预算，可以列出不等式：2x + y ≤ 10。 为了使苹果的数量最多，橙子的数量应尽可能少。 当x=0时，y=10，满足条件。 因此，小明最多可以买0个橙子和10个苹果。 小明有10块钱，想买橙子和苹果。橙子每个2块，苹果每个1块。我们需要计算他最多可以买多少个橙子和苹果。...

## Length effect (whole population, not just these rows)

- items under 100 B: **2,267**, median ratio **7.08**, max **114.1**
- items at or above 100 B: **8,154**, median ratio **2.46**, max **18.4**
- the short items are 0.7% of all bytes, so they barely move the byte-weighted 1.51x / 2.50x -- the distortion they cause is to a RANKED TABLE, which is why the two ranked groups have a floor and the random group does not.

## What the table does not show

The control wins **1 of 10,421** items (0.0%) in the full population, and 1 of the 12 rows here. A 12-row table cannot establish that share -- it is computed over everything and stated here so the table is not read as the population.
