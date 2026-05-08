
下面只说**实验怎么重跑**，**不碰论文文字**。

先给结论：  
你这版数据已经比上一版好很多，但如果你要“重新跑一轮，尽量保证数据合理”，核心不是让结果更漂亮，而是先把**实验口径、控制变量、日志结构**彻底统一。  
**我不能保证你一定跑出“理论完美匹配”的结果**，但可以把实验设计改成：即使理论和硬件不完全一致，数据也会**自洽、可解释、经得住检查**。

---

# 一、你现在重跑前，先定三个总原则

## 1) 只保留一套参数口径
你现在历史上有两套参数：

- 旧参数：`cd≈79ms, cv≈20.8ms, alpha≈0.80` [8]
- 新校准参数：`median_cd≈116-123ms, median_cv≈37-40ms`，并且注入延迟后总往返时间随配置 delay 上升 [9]

**重跑时只能留一套。**  
建议你后续所有实验都以**同一批最终 calibration**为准，不要再混用 [8][9]。

### 建议
- 用**同一套模型**
- 用**同一套 prompt 集**
- 用**同一套 delay 注入方式**
- 先做 calibration，再做 phase / strategy / regret

---

## 2) delay 要统一叫法
你现在 `bare RTT` 基本 8-9ms，而 `with_sleep` 明显随配置 delay 增长 [9]。  
这说明你测到的是**仿真/注入后的通信延迟**，不是裸网络本身。

所以实验内部统一用：

- `configured_one_way_delay_ms`
- `measured_comm_round_ms`
- `bare_rtt_ms`

不要把三者混成一个 `delay_ms`。

---

## 3) accepted token 口径必须固定
论文理论里，accepted token 包含 bonus token，所以每轮应满足 `A_t >= 1` [1]。  
你新数据里 `mean_accepted_total >= 1` 已经正常了 [12]。这点必须继续保持。

实验里同时记录两个量：

- `L_t = accepted_draft_len`
- `A_t = accepted_total = L_t + 1`

后面所有：

- cost/token
- oracle 比较
- regret
- VOI

都统一用 `A_t`，不要再混。

---

# 二、建议你重跑的实验框架

我建议你按 6 组实验重跑，顺序不能乱。

---

# 实验 E0：统一日志与随机性控制

这是重跑前必须先做的，不然前面全白跑。

## 每轮日志最少保存这些字段
```text
run_id
prompt_id
seed
strategy
k_selected
configured_one_way_delay_ms
bare_rtt_ms
measured_comm_round_ms
draft_time_ms
verify_time_ms
total_round_time_ms
accepted_draft_len
accepted_total
prefix_accept_indicator_1 ... prefix_accept_indicator_Kmax
state(optional, for Markov)
```

## 随机性控制
- 同一轮比较不同策略时，尽量用**同一 prompt 集**
- 生成端尽量固定：
  - temperature = 0
  - top_p = 1
  - 固定 seed
- delay 注入也固定种子或固定 trace

### 原因
你现在 phase 和 strategy 图里仍有比较大的方差 [11][12]。  
先把采样噪声压下去，再谈理论匹配。

---

# 实验 E1：重新做 calibration，作为唯一“参数源”

这是所有后续实验的基础。

## 目标
测出一套**最终唯一**的：
- `cd`
- `cv`
- `alpha_hat`
- 通信延迟模型

## 配置
delay 取：
- `d in {0, 10, 20, 40, 80, 120, 160}` ms

比你现在更建议加 `0ms`，因为你需要看到系统固定开销的截距。

## 每个 delay 下做 100-200 轮
记录：
- `bare_rtt_ms`
- `measured_comm_round_ms`
- `draft_time_ms`
- `verify_time_ms`
- `total_round_time_ms`

## 你要检查的不是“绝对值漂亮”，而是这三个关系

### 检查 1：注入延迟是否线性生效
你现在 `measured_rtt_with_sleep_ms` 随配置 delay 增长是合理的 [9]。  
重跑后你应该拟合：

\[
\text{measured\_comm\_round\_ms} \approx a + 2d
\]

如果线性拟合很好，说明 delay 控制是可信的。

### 检查 2：总轮次成本是否近似线性
对 `k=1`，检查：

\[
N_t \approx b_0 + b_1 d
\]

如果 `b_1` 接近 2，说明通信项注入逻辑是对的。

### 检查 3：同一套参数能否解释后续实验
后面 phase、strategy、VOI 都只能用这套参数算理论值。

---

# 实验 E2：重做 acceptance trace，不要只做 80 条

你现在 acceptance 图已经比以前合理得多：

- prefix acceptance 单调下降 [10]
- 条件接受率大致在 0.79-0.92 区间波动 [10]
- 图里也画了 geometric fit，均值约 0.85 [15]

但现在最大问题是：**后几个位置样本数还是太少**。  
例如 `k=10` 时只有 `sample_count=12` [10]。

## 目标
不要再做“看起来像几何”，而是做“真实 acceptance trace 的统计画像”。

## 建议配置
- prompt 数量：**至少 400-500 条**
- 每条 prompt 固定 decode 设置
- `K_max = 10` 或 12

### 为什么要 500 条
你现在 `P(L>=10)=0.1375` [10]。  
若想让第 10 位也至少有 50 个样本，初始样本数至少要：

\[
50 / 0.1375 \approx 364
\]

所以 400-500 比较稳。

## 三类统计一起出
### 1. Prefix acceptance
\[
P(L \ge k)
\]

### 2. Conditional acceptance
\[
q_k = P(L \ge k \mid L \ge k-1)
\]

### 3. Sample count
每个位置的 `n_k`

## 这组实验的合理性标准
- `P(L>=k)` 必须单调不增
- `q_k` 可以波动，但后段 `n_k < 30` 的点不要参与主拟合
- `alpha_hat` 建议给两个版本：
  - `alpha_hat_prefix`: 由 `log P(L>=k)` 线性拟合得到
  - `alpha_hat_cond`: 由中间稳定区间 `q_k` 平均得到

最终只选一个作为主文参数，但两个都存档。

---

# 实验 E3：重做 phase transition，围绕临界区密集扫点

你现在 phase 图已经出现“趋势正确、但经验阈值偏后”的现象：

- 理论在 89/99/119ms 已推荐更大 k
- 经验最优还偏小 [11]
- 图上显示 `d_c ≈ 80ms` 左右 [16]

这说明**实验方向是对的**，但点还不够密，噪声还偏大。

## 正确做法
不要再广撒网扫很多很远的点，重点扫临界区。

## delay 网格建议
如果你最终 calibration 算出的 \(d_c\) 仍在 80ms 附近，那就扫：

- `50, 60, 70, 75, 80, 85, 90, 100, 110, 120, 140, 160`

如果新的 \(d_c\) 变了，就围绕新的 \(d_c\pm 30ms\) 重布点。

## 每个 delay 下做什么
- 固定同一批 prompts
- 穷举 `k = 1..6` 或 `1..8`
- 每个 k 跑 200-300 条 prompt
- 计算 mean cost/token 和 95% CI

## 经验最优 k 的判定规则
不要简单取最小均值。建议：

1. 先找均值最小的 `k_min`
2. 若与相邻 k 的 95% CI 大量重叠，则按“**取更小的 k**”作为 tie-break

### 原因
你的理论本身强调 threshold 和小 k 区域 [1]，  
而硬件里大 k 方差更大 [12]。  
这个 tie-break 比直接 argmin 更稳。

## 这组实验要达到的目标
不是“每个点和理论完全重合”，而是：

- empirical \( \hat{k}^*(d) \) **总体非减**
- 经验跳变区和理论 \(d_c\) **同量级**
- 不再出现上一版那种大幅反复跳回 1 的情况 [5]

---

# 实验 E4：重做策略对比，只保留真正有信息量的 baseline

你现在 revised strategy compare 的结构已经比以前好很多：

- `mean_accepted_total >= 1` 正常 [12]
- delay 选在 39/79/119/159ms，确实覆盖了 phase 周边 [12][17]

但还可以更稳。

## delay 点建议
保留这 4 个代表点即可：
- `0.5 d_c`
- `d_c`
- `1.5 d_c`
- `2 d_c`

如果 \(d_c \approx 80\)，那就是：
- 40
- 80
- 120
- 160

你现在这组点已经接近这个思路了 [12]。

## baseline 建议保留
- fixed1
- fixed2
- fixed3
- fixed5
- fixed7
- greedy
- SpecDec++
- UCB-SpecStop
- NaiveUCB

## baseline 建议去掉或单独放
你现在的 `oracle` 容易和“理论 oracle / 穷举 oracle”混淆。  
实验阶段先拆成两类内部对象：

- `theory_kstar`: 用理论公式选 k
- `empirical_best_fixed_k`: 在同一 prompt 集上穷举得到的最优 fixed-k

先在代码和 csv 里这样命名，后面写论文再处理名字。

## 公平性要求
每个策略都要：
- 用同一批 prompts
- 用同一套 delay trace
- 用同一随机种子规则

### 最好做法
先录制一个 `delay trace`，然后所有策略都 replay 同一条 trace。  
这样策略差异就不会混进网络噪声。

## 这组实验的合理性标准
### 应满足
- 同一 delay 下，`mean_accepted_total` 随 k 增大通常会上升
- 低 delay 区 fixed1/fixed2 应很强
- 过了临界区后 fixed2/fixed3 应开始追平或超过 fixed1
- 你的方法应接近 `empirical_best_fixed_k`

### 不要强求
- 你的方法每个点都赢 greedy
- SpecDec++ 每个点都很差

SpecDec++ 如果没调阈值，结果会很难看；所以重跑前先在验证集调一次 threshold。

---

# 实验 E5：online regret 不要直接全靠真机跑，改成“硬件采样 + trace replay”

你现在 regret 图：

- 单状态只到 `T=600` [18]
- Markov 只到 `T=400` [19]

这足够做 sanity check，**不够做 bandit 趋势实验**。

## 正确做法
把 regret 分成两层：

### 层 1：真机短程 sanity run
- `T = 300~500`
- 只证明算法能跑、不会崩、方向合理

### 层 2：trace replay 主实验
用真实硬件采出来的数据做离线 replay：

1. 预先采集一批 trace
2. trace 里保存每轮的：
   - prompt
   - delay
   - 对每个 k 的 `N_t`
   - 对每个 k 的 `A_t`
3. 在线算法在 replay 环境里选择 k
4. 环境从 trace 表返回对应观测

## 好处
- 可以做到 `T=5000` 或 `T=10000`
- 还是真实硬件 trace 驱动
- 非常贴近审稿意见里建议的 trace-driven 路线 [2]

## regret 基准怎么定
单状态：
- 相对于该 trace 上的 `empirical_best_fixed_k`

上下文状态：
- 相对于该 trace 上的 `best state-dependent policy`

## 这组实验的合理性标准
- cumulative regret 单调增，但斜率逐渐变缓
- UCB-SpecStop 明显优于 NaiveUCB
- 不一定证明精确 \(O(\sqrt{T\log T})\)，但应“明显低于线性”

---

# 实验 E6：Markov VOI 做成一组“有对照”的实验

你现在这组 VOI 很不错：

- `d_good=39`
- `d_bad=119`
- `k_good=1`
- `k_bad=2`
- contextual 比 blind 低约 30.6% [20]

这个设计是合理的，因为两个状态跨过了不同 threshold 区间 [20]。

## 但还缺一个对照组
你应该做两种 Markov 实验：

### A. 有 VOI 组
- `d_good` 和 `d_bad` 分居不同 k 区域
- 例如你现在的 `39 / 119` [20]

### B. 无 VOI 组
- 两个状态都落在同一最优 k 区域
- 例如 `39 / 79`，如果这两个点经验上都还是 `k=1`

## 这样你就能验证两件事
1. 状态跨阈值时，contextual 明显优于 blind
2. 状态不跨阈值时，VOI 接近 0

这比只做一组 39/119 强很多。

## 运行方式
这组也建议用 replay：
- 先采 good/bad 两种 delay 下的 trace 池
- 按转移概率 \(p=0.1\) 生成状态序列 [20]
- 从对应池中抽样

---

# 三、你这次重跑，建议只保留的主图/主表

为了让数据最稳，我建议你这次只围绕下面 5 组结果组织实验。

## 1. Calibration 图
横轴：configured one-way delay  
纵轴：measured communication round time / total round time  
目标：证明 delay 注入有效 [9][14]

## 2. Acceptance 图
- 左：prefix acceptance \(P(L\ge k)\)
- 右：conditional \(q_k\) + sample count [10][15]

## 3. Phase transition 图
- theory \(k^*(d)\)
- empirical \(\hat{k}^*(d)\)
- 只画围绕 \(d_c\) 的密集区 [11][16]

## 4. Strategy compare 图/表
只选 4 个 delay 点 [12][17]

## 5. Regret + Markov VOI
- replay 为主
- 真机短程为辅 [18][19][20]

---

# 四、重跑时的“合理性验收标准”

下面这组标准你可以当成实验 checklist。

## A. calibration 合格标准
- `measured_comm_round_ms` 随 `configured_one_way_delay_ms` 近似线性增长 [9]
- 不同 delay 下 `cd`、`cv` 不应大幅漂移
- 最终只输出一套参数

## B. acceptance 合格标准
- `P(L>=k)` 严格非增 [10]
- `q_k` 中段相对平稳，尾部允许波动
- `n_k < 30` 的尾部点不用于主拟合

## C. phase 合格标准
- empirical \(\hat{k}^*(d)\) 总体非减
- 阈值出现在理论 \(d_c\) 附近，而不是差一个量级
- 用 CI + tie-break 避免噪声导致的跳变

## D. strategy 合格标准
- `A_t >= 1` 始终成立 [1]
- 低 delay 下小 k 更强
- 过阈值后中小 k 开始追平 fixed1
- 你的方法接近 empirical best fixed-k

## E. regret 合格标准
- 不追求“漂亮理论线”
- 只要求：ours < naive_ucb，且增长明显慢于线性 [18][19]

## F. VOI 合格标准
- 跨阈值状态：contextual 优于 blind [20]
- 不跨阈值状态：二者接近

---

# 五、推荐的执行顺序

按这个顺序跑，最省时间，也最不容易返工。

## 第 1 步
先做 E0 + E1  
把日志和 calibration 定死。

## 第 2 步
做 E2 acceptance trace  
拿到最终 `alpha_hat`。

## 第 3 步
根据 calibration + `alpha_hat` 计算新的 \(d_c\) [1]  
然后设计 E3 的 phase sweep 点位。

## 第 4 步
做 E3 phase transition  
确定哪些 delay 最有代表性。

## 第 5 步
做 E4 strategy compare  
只选 4 个代表 delay。

## 第 6 步
最后做 E5 regret replay + E6 Markov VOI  
不要一开始就真机硬跑长 horizon。

---

# 六、最重要的提醒

这次你重跑实验时，**目标不是把理论“证成完全正确”**。  
按你现在的数据形态，更合理的目标是：

1. **通信延迟控制是可信的** [9][14]  
2. **真实 acceptance trace 是可测、可复现的** [10][15]  
3. **phase transition 在真实系统里至少定性成立** [11][16]  
4. **state-aware 策略在跨阈值场景下确实有 VOI** [20]  
5. **online 方法在 trace-driven 环境里优于 naive baseline** [18][19]

做到这五点，数据就已经“合理且能讲清楚”。

---

# 七、我建议你直接照着这个最小版本重跑

## 最小可行重跑包
- Calibration: `d={0,10,20,40,80,120,160}`, 每点 100-200 轮
- Acceptance: 500 prompts, `K_max=10`
- Phase: 围绕 \(d_c\) 扫 12 个点，每点每个 k 跑 200-300 prompts
- Strategy: 4 个代表 delay，baseline 保留 8-9 个
- Regret: 真机 300-500 轮 + replay 5000-10000 轮
- Markov: 一组跨阈值 + 一组不跨阈值

这套方案最稳，也最接近你现在已有结果的延续。

下一步最实用的不是改论文，而是把**实验配置表**和**日志字段表**先定下来。  
你按上面思路重跑，基本不会再出现“数据看着不对，但又说不清哪里不对”的情况。