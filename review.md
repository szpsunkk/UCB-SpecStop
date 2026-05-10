下面是基于你**重新修改后的实验数据**的再分析。总体结论先说：

> **这版实验比上一版合理很多，尤其是 calibration、ratio-of-sums cost、phase transition 这几部分已经明显接近论文理论需要的实验形态。**  
> 但仍有两个关键问题不能直接放进论文：  
> 1. **strategy comparison 里的 empirical_oracle 仍然有定义或实现错误；**  
> 2. **Markov VOI 现在变成负数，不能支持论文中 value of information 的结论，需要重做。**

---

# 1. 总体判断

你这次修改后，实验质量有明显提升：

- 你已经把 cost 改成了 **ratio-of-sums**，这与论文中 UCB-SpecStop 的目标函数一致。论文明确指出应优化
  \[
  C(k)=\frac{\mathbb{E}[N_t|k]}{\mathbb{E}[A_t|k]}
  \]
  而不是逐轮平均 \(N_t/A_t\)，因为后者会产生 Jensen bias [1]。
- 你对每个 delay、每个 k 都做了独立测量，且每组有 300 rounds，这比之前只测少数 k 或样本不足更可靠 [2]。
- phase transition 的 empirical oracle k 已经呈现出非常清晰的单调增长：
  \[
  k^*=1 \rightarrow 2 \rightarrow 3 \rightarrow 4 \rightarrow 5
  \]
  这和论文 Theorem 2 中“最优 draft length 随通信延迟单调非降”的理论方向一致 [1][10]。
- 真实通信 round-trip 时间现在测得很稳定，基本符合：
  \[
  T_{\text{comm}}\approx 51\text{ms}+2\times \text{configured delay}
  \]
  这说明你现在的通信校准更可信 [2]。

但是，以下部分仍需修正：

- **strategy_compare 里的 empirical_oracle 在 74ms 和 149ms 明显不是最优。**
- **Markov VOI 为 -3.81%，这与论文中 VOI \(\ge 0\) 的理论定义冲突，说明 contextual policy 没有真正用 empirical optimal policy。**
- **acceptance profiling 仍只有 80 prompts，高 k 处样本仍然偏少。**

下面分模块详细说。

---

# 2. Calibration 实验：现在基本合理，可以保留

## 2.1 通信延迟测量合理

这版 calibration 最大的改进是：你现在对不同 k 和不同 delay 都测了通信 round。通信项非常稳定，并随 configured one-way delay 线性增加。例如：

| configured one-way delay | median comm round |
|---:|---:|
| 0ms | 约 51ms |
| 5ms | 约 62ms |
| 10ms | 约 71–74ms |
| 20ms | 约 91–93ms |
| 40ms | 约 131–133ms |
| 55ms | 约 161–164ms |
| 83ms | 约 217–220ms |
| 111ms | 约 275–276ms |
| 150ms | 约 353–355ms |

这说明真实通信代价不是简单的 \(2d\)，而更接近：

\[
T_{\text{comm}}(d) \approx 51\text{ms}+2d
\]

这非常重要。论文理论里用的是 \(2D\)，但真实系统里存在 RPC、HTTP、序列化、server handling、runtime scheduling 等固定开销。你的数据已经很好地证明了这一点 [2]。

论文中建议写成：

> In the hardware testbed, the effective communication cost is substantially larger than the injected network delay because it includes RPC, serialization, and runtime overhead. Therefore, we use the measured round-trip communication time rather than the configured one-way delay.

---

## 2.2 时间分解现在比之前一致很多

之前的问题是 total round time 与 draft、verify、communication 无法加和。现在看，很多配置下：

\[
T_{\text{total}} \approx T_{\text{draft}} + T_{\text{comm}}
\]

例如 configured delay = 0, k = 1：

- median cd per token ≈ 110.67ms
- median comm round ≈ 51.09ms
- median total round time ≈ 161.91ms

两者加起来：

\[
110.67+51.09=161.76\text{ms}
\]

和 total 161.91ms 几乎一致 [2]。

再看 configured delay = 55, k = 3：

- median cd per token ≈ 89.90ms
- k = 3，所以 draft time 约 \(3\times89.90=269.70\)ms
- median comm round ≈ 162.48ms
- 总和约 \(432.18\)ms
- median total round time ≈ 433.23ms [2]

也非常接近。

这说明你现在的测量口径基本修正了。建议论文硬件实验中使用：

\[
T_{\text{round}}(k,d)
=
T_{\text{draft}}(k)
+
T_{\text{rpc}}(k,d)
\]

而不要强行写成：

\[
k c_d + 2d + (k+1)c_v
\]

因为在你的真实系统里，server verification 已经包含在 RPC round 中了。

---

## 2.3 draft time 和 verify time 的趋势合理，但说明真实系统非线性

你的 measured draft per token 随 k 增大而下降。例如 delay = 0 时：

| k | median cd per token |
|---:|---:|
| 1 | 110.67ms |
| 2 | 82.40ms |
| 3 | 78.09ms |
| 5 | 74.28ms |
| 10 | 72.49ms |

这说明 draft model 推理中存在固定 overhead，k 越大，per-token 平均 overhead 越低 [2]。

server 端也类似。server_total 大约一直在 39–40ms，而 cv per token 会随 k 下降。这说明 target verification 的主要开销更接近一个 batch verification round，而不是严格的 \((k+1)c_v\) 线性增长 [2]。

因此硬件实验部分建议不要用论文理论里的线性 cost model 直接拟合硬件，而是写：

\[
\widehat{C}(k,d)
=
\frac{
\sum_r T_r(k,d)
}{
\sum_r A_r(k,d)
}
\]

这与你现在的 cost_ratio_of_sums 一致。

---

# 3. Acceptance 实验：趋势合理，但样本仍偏少

你的新版 acceptance 数据是：

| k | \(P(L\ge k)\) |
|---:|---:|
| 1 | 0.6125 |
| 2 | 0.4625 |
| 3 | 0.3875 |
| 4 | 0.3125 |
| 5 | 0.275 |
| 6 | 0.25 |
| 7 | 0.1875 |
| 8 | 0.1625 |
| 9 | 0.1375 |
| 10 | 0.1125 |

整体单调下降，这是合理的 [3]。

相比上一版，这组 acceptance 明显更好：之前 \(P(L\ge1)=0.475\)，现在变成 0.6125，说明 draft-target pair 的接受率更高，或者你的 verify/sample 设置更稳定 [3]。

但是 conditional acceptance 仍然不是常数：

| k | conditional \(q_k\) |
|---:|---:|
| 1 | 0.6125 |
| 2 | 0.7551 |
| 3 | 0.8378 |
| 4 | 0.8065 |
| 5 | 0.88 |
| 6 | 0.9091 |
| 7 | 0.75 |
| 8 | 0.8667 |
| 9 | 0.8462 |
| 10 | 0.8182 |

这说明真实 acceptance 仍不符合论文主理论里的 single-\(\alpha\) geometric assumption。论文中 Assumption 1 假设连续接受 k 个 token 的概率是 \(\alpha^k\)，这是为了推导 closed-form threshold 和 logarithmic scaling [1]。

所以论文里应当明确区分：

- simulation：使用 geometric acceptance，验证理论；
- hardware：使用 empirical prefix acceptance curve，做真实系统验证。

建议你在论文中写：

> The geometric acceptance model is used for tractable theoretical analysis. In the hardware experiment, the empirical prefix acceptance curve is position-dependent; therefore we compute hardware costs and empirical oracle policies using measured accepted tokens directly.

此外，当前 acceptance 仍只有 80 prompts [9]。高 k 的 sample_count 仍然少，例如 k=10 只有 11 个有效样本 [3]。正式论文最好增加到：

\[
n_{\text{prompts}}\ge 500
\]

如果时间紧，至少扩到 200–300。

---

# 4. Phase transition 实验：这部分现在最有价值

你新版 phase summary 非常有用。empirical oracle k 如下：

| configured delay | effective one-way delay | empirical oracle k | empirical oracle cost |
|---:|---:|---:|---:|
| 0ms | 5.61ms | 1 | 101.87 |
| 5ms | 10.65ms | 1 | 112.24 |
| 20ms | 25.64ms | 1 | 131.98 |
| 40ms | 45.91ms | 2 | 156.24 |
| 55ms | 60.95ms | 2 | 171.58 |
| 83ms | 89.13ms | 3 | 195.57 |
| 111ms | 117.72ms | 4 | 216.67 |
| 150ms | 156.88ms | 5 | 243.37 |

这非常好，因为它清楚显示：

\[
k^*(d)
\text{ 随 delay 单调非降}
\]

这直接支持论文的 Theorem 2，即 optimal draft length 随 communication delay 单调不下降 [1][10]。

这部分可以作为硬件实验的核心结果。

---

## 4.1 但理论预测和 empirical oracle 仍有偏差

你的 summary 中有三类 theory：

| delay | k_theory_geometric | k_theory_calibrated | k_theory_empirical | k_empirical_oracle |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1 |
| 5 | 1 | 1 | 1 | 1 |
| 20 | 1 | 1 | 1 | 1 |
| 40 | 1 | 1 | 1 | 2 |
| 55 | 1 | 1 | 1 | 2 |
| 83 | 1 | 2 | 1 | 3 |
| 111 | 2 | 2 | 1 | 4 |
| 150 | 3 | 3 | 2 | 5 |

可以看到：

- geometric theory 预测的 k 偏小；
- calibrated geometric theory 有改进，但仍偏小；
- empirical-prefix theory 反而更保守，也偏小；
- 真实 empirical oracle 增长更快 [10]。

这说明真实硬件 cost model 中，长 k 的 amortization 收益比理论模型估计得更强。主要原因可能是：

1. draft per-token cost 随 k 增大下降；
2. server verify cost 几乎是固定 round cost，而不是线性 \((k+1)c_v\)；
3. communication 有很大固定开销，增加 k 可以更有效 amortize；
4. empirical acceptance 比单一 \(\alpha\) 模型复杂。

所以论文中不能说：

> Hardware exactly confirms the closed-form critical delay.

应该说：

> Hardware confirms the qualitative monotonic threshold behavior, but the exact transition points are shifted by non-linear draft cost, batched verification, RPC overhead, and non-geometric acceptance.

这是很合理、也很容易被审稿人接受的表述。

---

## 4.2 详细 cost curve 也很合理

以 delay = 111ms 为例，详细 phase 数据显示：

| k | cost_ratio_of_sums |
|---:|---:|
| 1 | 245.27 |
| 2 | 226.57 |
| 3 | 217.74 |
| 4 | 216.67 |
| 5 | 219.15 |
| 7 | 229.42 |
| 10 | 258.95 |

这是非常漂亮的 U-shaped curve，最优 k=4 [4]。

以 delay = 150ms 为例：

| k | cost_ratio_of_sums |
|---:|---:|
| 1 | 292.38 |
| 2 | 263.63 |
| 3 | 249.47 |
| 4 | 244.49 |
| 5 | 243.37 |
| 7 | 250.90 |
| 10 | 278.99 |

同样是 U-shaped，最优 k=5 [4]。

这说明你的硬件实验已经可以很好地支持论文图示中的 U-shaped cost curve 叙事。

建议论文中增加一张硬件版 figure：

- x-axis：k；
- y-axis：cost_ratio_of_sums；
- curves：delay = 0, 40, 83, 111, 150；
- 标出每条曲线的 empirical oracle k。

这张图会非常有说服力。

---

# 5. Strategy comparison：改进明显，但 empirical_oracle 仍有 bug

这部分还有问题，不能直接放论文。

你现在 strategy comparison 已经改成了 cost_ratio_of_sums，这一点是对的 [5][6]。但是 empirical_oracle 仍然不是总是最优。

---

## 5.1 delay = 37ms：基本合理

configured delay = 37ms 时：

| strategy | cost |
|---|---:|
| greedy | 159.97 |
| calibrated geometric oracle | 161.43 |
| fixed2 | 161.48 |
| fixed3 | 161.53 |
| empirical oracle | 161.57 |
| fixed1 | 161.79 |

这里各策略差距很小。greedy 比 empirical oracle 低约 1%，这可能是采样噪声或策略运行 trace 不完全 paired 所致 [5][6]。

这个结果可以接受，但论文里不要强调 oracle 绝对最优。可以写 fixed1/fixed2/fixed3 在低延迟区间差异不大。

---

## 5.2 delay = 74ms：empirical_oracle 明显错误

configured delay = 74ms 时：

| strategy | cost |
|---|---:|
| fixed3 | 195.93 |
| fixed2 | 197.93 |
| fixed4 | 198.51 |
| ucb | 203.90 |
| fixed5 | 205.17 |
| empirical_oracle | 209.43 |

这里 fixed3 是最优，但 empirical_oracle 却是 209.43，而且 k_used = 1 [5][6]。

这说明 strategy comparison 里的 empirical_oracle 不是“从同一批 fixed-k 测量结果中选最小 cost 的 k”。它可能是：

1. 用了另一个脚本预估的 oracle；
2. 用了 empirical prefix theory，而不是真实 measured cost；
3. 使用了错误的 delay-to-k 映射；
4. 使用了旧的 oracle table；
5. 没有和 fixed-k 使用相同 trace/prompt。

如果叫 empirical_oracle，它必须满足：

\[
C_{\text{emp oracle}}(d)
=
\min_k C_{\text{fixed }k}(d)
\]

在 74ms 下它应该选择 k=3，而不是 k=1。

---

## 5.3 delay = 111ms：接近合理

configured delay = 111ms 时：

| strategy | cost |
|---|---:|
| fixed3 | 232.31 |
| empirical_oracle | 234.14 |
| fixed4 | 234.78 |
| fixed2 | 237.64 |
| calibrated oracle | 238.99 |
| ucb | 240.67 |

这里 empirical_oracle 使用 k=4，成本 234.14；fixed3 更低 232.31 [5][6]。差距约 0.8%，可以认为在 200 rounds 下有采样误差。

但是严格来说，empirical_oracle 仍应该选择 fixed3。建议增加 n_rounds 到 500 或 1000，并使用 paired prompts。

---

## 5.4 delay = 149ms：empirical_oracle 再次明显错误

configured delay = 149ms 时：

| strategy | cost |
|---|---:|
| fixed5 | 250.96 |
| fixed4 | 252.00 |
| calibrated oracle | 257.49 |
| specdec_pp | 258.29 |
| ucb | 259.55 |
| empirical_oracle | 272.28 |

这里 fixed5 是最优，而 empirical_oracle 选择了 k=2，成本 272.28 [5][6]。

这和 phase summary 中 delay=150ms 的 empirical oracle k=5 也矛盾。phase summary 说 150ms 下 k_empirical_oracle=5，cost=243.37 [10]；但 strategy comparison 里 149ms 的 empirical_oracle 却是 k=2，cost=272.28 [5]。

这说明 strategy_compare 的 empirical_oracle 实现仍有问题。

---

## 5.5 建议如何修正 strategy comparison

你应该把 strategy comparison 中的 oracle 分成三个名字：

### 1. offline measured oracle

这个是最重要的，定义为：

\[
k^*_{\text{offline}}(d)
=
\arg\min_k
\frac{\sum_r T_r(k,d)}{\sum_r A_r(k,d)}
\]

它必须从 fixed-k 结果中直接取最小值。

例如：

- delay 37ms：offline oracle 应该大约是 greedy 或 fixed2/fixed3；
- delay 74ms：offline oracle 应该是 fixed3；
- delay 111ms：offline oracle 应该是 fixed3；
- delay 149ms：offline oracle 应该是 fixed5 [6]。

### 2. calibrated geometric oracle

这个可以保留，表示用拟合 \(\alpha\)、measured cost model 预测的 k。

### 3. empirical-prefix predicted oracle

这个也可以保留，但不能叫 empirical oracle。它只是“用 empirical prefix curve 预测的 oracle”，不是 measured oracle。

建议表格列改成：

| strategy | 37 | 74 | 111 | 149 |
|---|---:|---:|---:|---:|
| fixed1 | ... | ... | ... | ... |
| fixed2 | ... | ... | ... | ... |
| fixed3 | ... | ... | ... | ... |
| fixed4 | ... | ... | ... | ... |
| fixed5 | ... | ... | ... | ... |
| UCB | ... | ... | ... | ... |
| SpecDec++ | ... | ... | ... | ... |
| calibrated-geometric oracle | ... | ... | ... | ... |
| empirical-prefix predicted | ... | ... | ... | ... |
| offline measured oracle | min | min | min | min |

这样就不会出现 oracle 比 fixed-k 差的问题。

---

# 6. UCB 实验：有进步，但还不能完全支撑 regret theorem

你现在有 arm pull diagnostics，显示 UCB 在 5000 rounds 中快速集中到某个 k。例如 UCB 到 t=5000 时，k3 被拉了 4942 次，k4 被拉 35 次，其他 arm 基本只拉 1–15 次 [11]。

这说明 UCB 确实在收敛，并且主要选择 k=3 [11]。

但这里有两个问题。

---

## 6.1 还缺 cumulative regret 曲线

论文中 UCB-SpecStop 的核心理论是：

\[
O(\sqrt{K_{\max}T\log T})
\]

regret bound [1]。

你现在提供的是 arm pull diagnostics，而不是 cumulative regret curve。它能说明 UCB 会集中选择某个 arm，但不能证明 regret 是 sublinear。

正式论文里需要输出：

```text
t
selected_k
round_cost
accepted_tokens
estimated_C(k)
oracle_C
instant_regret
cumulative_regret
```

然后画：

1. cumulative regret vs t；
2. log-log plot；
3. slope 是否接近 1/2；
4. selected arm distribution。

---

## 6.2 UCB 探索可能不足

从 diagnostics 看，UCB 对很多 arm 只拉了一次：

```text
k1=1, k2=1, k5=1, k6=1, k8=1, k10=1
```

而 k3 被拉了 4942 次 [11]。

如果 k3 真的是最优，这当然没问题。但从理论 UCB 的角度看，只拉一次就永久淘汰某些 arm，有点危险。可能原因是：

- 初始单次样本 cost 很高；
- confidence bonus 不够大；
- cost scale 没有 normalization；
- \(\beta\) 太小；
- \(S_A(k)\) 大小导致 confidence width 过快变窄。

建议：

1. 每个 arm 先强制 warm-up 5–10 次；
2. 对 cost 做 normalization；
3. 调大 \(\beta\)；
4. 输出每个 arm 的 confidence lower bound；
5. 确认最优 arm 是否确实是 k3。

否则审稿人可能会质疑：这不是 UCB 的有效探索，而是初始样本偶然导致的 premature convergence。

---

# 7. Markov VOI：当前结果不能用，需要重做

这是新版实验中最明显的问题。

你现在 Markov VOI 是：

```json
blind_cost_ratio_of_sums = 208.51
contextual_cost_ratio_of_sums = 216.46
voi_pct = -3.81%
d_good = 37
d_bad = 111
k_good_empirical = 1
k_bad_empirical = 1
k_good_geometric = 1
k_bad_geometric = 2
```

也就是说，contextual policy 反而比 blind policy 更差，VOI 是负数 [8]。

但论文 Theorem 6 定义的 VOI 是：

\[
VOI =
C(k^*_{\text{blind}},\mu_D)
-
\mathbb{E}_s[C(k^*(s),d(s))]
\ge 0
\]

因为 state-dependent oracle 至少可以模仿 blind policy，所以理论上不会更差 [1]。

因此负 VOI 说明实验实现有问题，不是理论被推翻。

---

## 7.1 为什么会出现负 VOI？

从你的 JSON 看，原因很可能是：

- empirical policy 认为 good state 和 bad state 都应该用 k=1；
- 但 contextual policy 可能用了 geometric policy，即 good 用 k=1，bad 用 k=2；
- 而在这组 trace 上，k=2 并不比 k=1 好，所以 contextual 反而变差 [8]。

也就是说，你现在混用了：

- geometric oracle；
- empirical oracle；
- measured cost；
- Markov replay。

这会导致 VOI 变负。

---

## 7.2 如何修正 Markov VOI

你需要定义清楚两种 VOI：

### A. empirical contextual VOI

这个应该使用每个 state 的 measured empirical oracle：

\[
k^*_{\text{emp}}(s)
=
\arg\min_k
\widehat{C}(k,d_s)
\]

如果 d_good=37, d_bad=111，根据 strategy_compare，大致应该是：

- d=37：k=2 或 k=3 都差不多；
- d=111：k=3 最好 [6]。

所以 empirical contextual policy 不应该是 k_good=1, k_bad=1。

你可以从 phase summary 选择更稳的状态：

- good = 20ms，empirical oracle k=1；
- bad = 150ms，empirical oracle k=5 [10]。

这样更容易得到正 VOI。

---

### B. geometric contextual VOI

如果你想验证论文 closed-form theorem，可以用 geometric theory policy。但那就必须和 geometric cost model 一起算，而不能拿 geometric k 去跑真实 measured cost 后再声称 VOI 应该非负。

换句话说：

- theoretical VOI：用理论模型算；
- empirical VOI：用 measured oracle 算。

不要混用。

---

## 7.3 当前 Markov VOI 不建议放论文

现在这个结果是：

\[
VOI=-3.81\%
\]

它不能支持论文结论 [8]。

建议暂时不要放，除非你在论文里把它作为反例说明：

> A geometric policy may be suboptimal on real hardware if not calibrated.

但如果你想支持 Theorem 6，就必须重做。

---

# 8. 论文实验部分现在应如何写

基于新版数据，我建议你硬件实验部分这样组织。

---

## 8.1 Hardware calibration

可以写：

> We first calibrate the real edge-cloud system. The measured communication round-trip time increases almost linearly with the injected one-way delay but contains a non-negligible fixed RPC overhead of approximately 50 ms.

然后报告表格：

| configured delay | comm round |
|---:|---:|
| 0 | 51ms |
| 20 | 91ms |
| 55 | 162ms |
| 83 | 219ms |
| 111 | 275ms |
| 150 | 354ms |

并说明 total round time 近似等于 draft generation time 加 RPC round time [2]。

---

## 8.2 Empirical acceptance

写：

> The empirical prefix acceptance curve decreases with k but is not perfectly geometric.

然后报告：

\[
P(L\ge1)=0.6125,\quad
P(L\ge5)=0.275,\quad
P(L\ge10)=0.1125
\]

并说明 conditional acceptance varies across positions [3]。

---

## 8.3 Empirical phase transition

这是你的亮点。可以写：

> The empirical oracle draft length increases monotonically with delay, from k=1 at low delay to k=5 at 150 ms configured one-way delay.

报告表：

| delay | empirical oracle k |
|---:|---:|
| 0 | 1 |
| 5 | 1 |
| 20 | 1 |
| 40 | 2 |
| 55 | 2 |
| 83 | 3 |
| 111 | 4 |
| 150 | 5 |

这非常支持论文的 delay-monotone threshold structure [1][10]。

但要补一句：

> The exact transition boundary differs from the closed-form prediction because the hardware system exhibits non-linear draft cost, batched verification, and non-geometric acceptance.

---

## 8.4 Strategy comparison

这部分需要先修正 oracle，再放。

你可以暂时只放 fixed-k 和 UCB，不放 empirical_oracle。或者放的时候把 oracle 改成：

\[
\min_k \text{fixed-k measured cost}
\]

例如从 table_ii_revised 中可得到：

| delay | best fixed-k | best fixed cost |
|---:|---:|---:|
| 37 | greedy/fixed2/fixed3 接近 | ≈160–162 |
| 74 | fixed3 | 195.93 |
| 111 | fixed3 | 232.31 |
| 149 | fixed5 | 250.96 |

然后再比较 UCB：

| delay | UCB cost |
|---:|---:|
| 37 | 172.57 |
| 74 | 203.90 |
| 111 | 240.67 |
| 149 | 259.55 |

可以说：

> UCB approaches the best fixed-k baseline but still has a small exploration overhead under the 200-round budget.

但如果要和论文 regret theorem 对齐，建议用 5000 rounds 的 UCB 结果。

---

# 9. 你现在还需要修改的地方

## 必须修改 1：strategy_compare 的 empirical_oracle

当前 empirical_oracle 不是 measured oracle。你需要重算：

```python
offline_oracle_cost[d] = min(cost_ratio_of_sums[fixed_k, d])
offline_oracle_k[d] = argmin_k(cost_ratio_of_sums[fixed_k, d])
```

不要单独 rerun 一个 oracle policy，除非它严格使用同一批 prompts 和同一批 trace。

---

## 必须修改 2：Markov VOI

当前 VOI 为负，不能用。重做时：

1. 用 measured empirical oracle k；
2. good/bad states 选能跨越 empirical threshold 的延迟；
3. 不要混用 geometric k 和 empirical cost；
4. contextual policy 至少要允许选择 blind k，否则 VOI 不可能为负。

推荐配置：

```text
good delay = 20ms, empirical k = 1
bad delay = 150ms, empirical k = 5
p_g2b = p_b2g = 0.1
```

然后比较：

- blind fixed oracle；
- contextual empirical oracle；
- contextual UCB；
- non-contextual UCB。

---

## 必须修改 3：acceptance 样本量

当前 n_prompts=80 [9]。建议至少：

```text
n_prompts >= 300
最好 >= 500
```

尤其是你要报告 \(P(L\ge10)\) 时，80 个 prompt 太少。

---

## 必须修改 4：UCB 输出 regret curve

现在 diagnostics 可以说明 arm selection，但不能验证 regret theorem。需要导出：

```text
t, selected_k, N_t, A_t, C_hat, oracle_C, cumulative_regret
```

并画 log-log slope。

---

# 10. 最终结论

这版实验已经比上一版强很多。可以认为：

## 可以保留并写入论文的部分

1. **Calibration**：通信 round-trip 随 injected delay 线性增加，且存在约 50ms 固定 RPC overhead [2]。
2. **Timing decomposition**：total round time 现在基本可以由 draft time + RPC time 解释 [2]。
3. **Acceptance curve**：真实 prefix acceptance 单调下降但非 geometric [3]。
4. **Empirical phase transition**：empirical oracle k 随 delay 从 1 增加到 5，强力支持 delay-monotone threshold 结论 [10]。
5. **U-shaped cost curve**：111ms 和 150ms 下的 cost-vs-k 曲线非常漂亮 [4]。

## 不能直接放论文的部分

1. **strategy comparison 中的 empirical_oracle**：当前不是 measured best k，必须重算 [5][6]。
2. **Markov VOI**：当前为负，说明实现或定义混乱，不能支持论文 Theorem 6 [8]。
3. **UCB regret**：还缺 cumulative regret curve，不能直接验证 \(O(\sqrt{T\log T})\) [1][11]。

## 论文中最稳妥的表述

你可以这样写：

> The hardware experiment confirms the qualitative threshold structure predicted by the theory: the empirically optimal draft length increases monotonically with communication delay. However, the exact transition points differ from the closed-form geometric model because the real system exhibits non-geometric acceptance, batched verification, and substantial RPC overhead. Therefore, deployment should use measured communication cost and empirical accepted-token statistics for calibration.

这个表述既符合你的理论，也符合现在的数据。