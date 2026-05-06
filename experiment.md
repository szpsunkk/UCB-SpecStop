# 完整实验方案设计

## 一、实验总体架构

根据论文的理论框架 [1]，实验需要验证五个核心理论预测：Phase Transition（Theorem 5）、Monotonicity（Theorem 2）、Latency Improvement（Corollary 1）、UCB-SpecStop收敛性（Theorem 7）、VOI（Theorem 6）。实验分**理论仿真**和**硬件验证**两个阶段。

---

## 二、Baseline选择

### 需要对比的方法（共6类）

| 编号 | Baseline | 来源 | 说明 |
|------|----------|------|------|
| B1 | **Fixed-$k$ (k=1,3,5,7,10)** | 标准做法 [1] | 最常见的工程实践，固定draft长度不随网络变化 |
| B2 | **Greedy（无通信感知）** | Leviathan et al. [1] | 忽略通信代价，用本地centralized最优$k$，即$d=0$时的$k^*$ |
| B3 | **SLED-style timeout** | SLED [14] | 设置固定超时阈值，超时则缩短draft；未超时则增长draft |
| B4 | **SpecDec++ threshold** | SpecDec++ [15] | 基于acceptance概率阈值$p_{\min}$：当$\alpha^k < p_{\min}$时停止 |
| B5 | **Oracle optimal $k^*(d)$** | 本文Theorem 5 | 已知所有参数时的理论最优，作为性能上界 |
| B6 | **UCB1 (naive ratio)** | Auer et al. [19] | 标准UCB1直接优化$\mathbb{E}[N_t/A_t]$，验证ratio-of-sums的必要性 |

### 各Baseline实现细节

```python
# B1: Fixed-k
def fixed_k_policy(k_fixed):
    return k_fixed  # 始终返回固定值

# B2: Greedy (d=0 optimal)
def greedy_policy(alpha, cd, cv):
    # 忽略通信代价，用d=0计算k*
    return compute_kstar(alpha, cd, cv, d=0)

# B3: SLED-style timeout
def sled_policy(k_current, last_rtt, timeout_threshold):
    if last_rtt > timeout_threshold:
        return max(1, k_current - 1)  # 缩短
    else:
        return k_current + 1  # 增长

# B4: SpecDec++ threshold
def specdec_pp_policy(alpha, p_min=0.3):
    # 停在 alpha^k >= p_min 的最大k
    k = 1
    while alpha**(k+1) >= p_min:
        k += 1
    return k

# B5: Oracle
def oracle_policy(alpha, cd, cv, d):
    return compute_kstar(alpha, cd, cv, d)  # 公式(9)+(5)

# B6: Naive UCB1
# 与Algorithm 1相同，但用 mean(N_t/A_t) 代替 S_N/S_A
```

---

## 三、理论仿真实验

### 实验S1：Phase Transition验证

**目标：** 验证Theorem 5中的临界延迟$d_c$和对数缩放律 [1]

```python
import numpy as np
import matplotlib.pyplot as plt

def B(k, alpha):
    return (1 - alpha**(k+1)) / (1 - alpha)

def C(k, d, alpha, cd, cv):
    N = k * (cd + cv) + 2 * d + cv
    return N / B(k, alpha)

def compute_kstar(alpha, cd, cv, d, K_max=20):
    costs = [C(k, d, alpha, cd, cv) for k in range(1, K_max+1)]
    return np.argmin(costs) + 1

def dc_theory(alpha, cd, cv):
    return (cd + cv) * (1 + alpha) / (2 * alpha**2) \
           - (cd + 2 * cv) / 2

# 参数
cd, cv = 1.0, 0.5  # ms
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
d_range = np.arange(0, 500, 0.5)  # ms

fig, ax = plt.subplots(figsize=(8, 5))
for alpha in alphas:
    kstars = [compute_kstar(alpha, cd, cv, d) for d in d_range]
    ax.plot(d_range, kstars, label=f'α={alpha}')
    
    # 对数包络线 (Theorem 5)
    dc = dc_theory(alpha, cd, cv)
    envelope = np.log(2 * d_range[d_range > dc] * (1 - alpha) 
                      / (cd + cv)) / np.log(1/alpha) - 1
    ax.plot(d_range[d_range > dc], envelope, '--', 
            color='gray', alpha=0.5)
    
    # 标注dc
    ax.axvline(x=dc, linestyle=':', alpha=0.3)

ax.set_xlabel('Mean delay d (ms)')
ax.set_ylabel('Optimal draft length k*')
ax.legend()
ax.set_title('Phase Transition and Logarithmic Scaling')
plt.savefig('fig_phase_transition.pdf', bbox_inches='tight')
```

**输出：** Fig. 2，验证阶梯状$k^*(d)$曲线和对数包络

**关键观测点：**
- 报告每个$\alpha$的理论$d_c$值与实测跳变点的误差
- $\alpha=0.7$时$d_c \approx$多少ms

### 实验S2：Latency对比表

**目标：** 填充Table I，验证optimal $k^*$相对fixed-$k$的改进 [1]

```python
# 参数
alpha = 0.7
cd, cv = 1.0, 0.5
delays = [5, 10, 30, 50, 100, 200, 500]

print("d(ms) | k* | k=1  | k=3  | k=5  | k=7  | k=10 | "
      "Greedy | SLED | SpecDec++ | Improve%")
print("-" * 90)

for d in delays:
    kstar = compute_kstar(alpha, cd, cv, d)
    c_opt = C(kstar, d, alpha, cd, cv)
    
    results = {}
    for k in [1, 3, 5, 7, 10]:
        results[k] = C(k, d, alpha, cd, cv)
    
    # B2: Greedy (d=0)
    k_greedy = compute_kstar(alpha, cd, cv, d=0)
    c_greedy = C(k_greedy, d, alpha, cd, cv)
    
    # B4: SpecDec++ (p_min=0.3)
    k_spec = 1
    while alpha**(k_spec+1) >= 0.3:
        k_spec += 1
    c_spec = C(k_spec, d, alpha, cd, cv)
    
    # 相对k=5的改进
    improve = (results[5] - c_opt) / results[5] * 100
    
    print(f"{d:5d} | {kstar:2d} | {results[1]:.2f} | "
          f"{results[3]:.2f} | {results[5]:.2f} | "
          f"{results[7]:.2f} | {results[10]:.2f} | "
          f"{c_greedy:.2f} | -- | {c_spec:.2f} | "
          f"{improve:.1f}%")
```

**输出：** 扩展的Table I（含所有baseline），验证Abstract中"38%"的数字

### 实验S3：UCB-SpecStop收敛性

**目标：** 验证Theorem 7的regret bound [1]

```python
def ucb_specstop(T, K_max, alpha, cd, cv, d_mean, d_std, 
                 beta=1.0, n_trials=1000):
    """Monte Carlo simulation of Algorithm 1"""
    all_regrets = np.zeros((n_trials, T))
    
    # 计算oracle最优
    kstar = compute_kstar(alpha, cd, cv, d_mean)
    c_star = C(kstar, d_mean, alpha, cd, cv)
    
    for trial in range(n_trials):
        S_N = np.zeros(K_max)
        S_A = np.zeros(K_max)
        T_k = np.zeros(K_max)
        cumulative_regret = 0.0
        
        for t in range(1, T + 1):
            # 选择arm
            if np.any(T_k == 0):
                k = np.where(T_k == 0)[0][0]
            else:
                indices = S_N / S_A - beta * np.sqrt(
                    T_k * np.log(t)) / S_A
                k = np.argmin(indices)
            
            # 模拟一轮
            k_val = k + 1  # arms are 1-indexed
            D_t = np.random.exponential(d_mean)
            N_t = k_val * (cd + cv) + 2 * D_t + cv
            
            # 模拟acceptance
            n_accepted = 0
            for i in range(k_val):
                if np.random.random() < alpha:
                    n_accepted += 1
                else:
                    break
            A_t = n_accepted + 1  # bonus token
            
            # 更新统计
            S_N[k] += N_t
            S_A[k] += A_t
            T_k[k] += 1
            
            # 计算regret
            c_k = C(k_val, d_mean, alpha, cd, cv)
            cumulative_regret += (c_k - c_star)
            all_regrets[trial, t-1] = cumulative_regret
    
    return all_regrets

# 运行
T = 10000
K_max = 20
alpha = 0.7
cd, cv = 1.0, 0.5

# 场景1: 大gap (d=100ms, 容易区分)
regrets_large = ucb_specstop(T, K_max, alpha, cd, cv, 
                              d_mean=100, d_std=20)

# 场景2: 小gap (d=10ms, 难以区分)
regrets_small = ucb_specstop(T, K_max, alpha, cd, cv, 
                              d_mean=10, d_std=5)

# 同时运行B6: Naive UCB1
def naive_ucb1(T, K_max, alpha, cd, cv, d_mean, beta=1.0, 
               n_trials=1000):
    """标准UCB1优化E[N/A]而非E[N]/E[A]"""
    all_regrets = np.zeros((n_trials, T))
    kstar = compute_kstar(alpha, cd, cv, d_mean)
    c_star = C(kstar, d_mean, alpha, cd, cv)
    
    for trial in range(n_trials):
        sum_ratio = np.zeros(K_max)
        T_k = np.zeros(K_max)
        cumulative_regret = 0.0
        
        for t in range(1, T + 1):
            if np.any(T_k == 0):
                k = np.where(T_k == 0)[0][0]
            else:
                mean_ratio = sum_ratio / T_k
                indices = mean_ratio - beta * np.sqrt(
                    np.log(t) / T_k)
                k = np.argmin(indices)
            
            k_val = k + 1
            D_t = np.random.exponential(d_mean)
            N_t = k_val * (cd + cv) + 2 * D_t + cv
            n_accepted = 0
            for i in range(k_val):
                if np.random.random() < alpha:
                    n_accepted += 1
                else:
                    break
            A_t = n_accepted + 1
            
            sum_ratio[k] += N_t / A_t
            T_k[k] += 1
            
            c_k = C(k_val, d_mean, alpha, cd, cv)
            cumulative_regret += (c_k - c_star)
            all_regrets[trial, t-1] = cumulative_regret
    
    return all_regrets

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
t_range = np.arange(1, T + 1)

# 参考线: O(sqrt(t log t))
ref = 5 * np.sqrt(t_range * np.log(t_range + 1))

for ax, regrets, regrets_naive, title in [
    (axes[0], regrets_large, 
     naive_ucb1(T, K_max, alpha, cd, cv, 100), 
     'd=100ms (large gap)'),
    (axes[1], regrets_small, 
     naive_ucb1(T, K_max, alpha, cd, cv, 10), 
     'd=10ms (small gap)')]:
    
    ax.plot(t_range, np.mean(regrets, axis=0), 
            label='UCB-SpecStop (ours)')
    ax.plot(t_range, np.mean(regrets_naive, axis=0), 
            label='Naive UCB1 (B6)', linestyle='--')
    ax.plot(t_range, ref, ':', color='gray', 
            label=r'$O(\sqrt{t\log t})$')
    ax.set_xlabel('Round t')
    ax.set_ylabel('Cumulative Regret R(t)')
    ax.set_title(title)
    ax.legend()

plt.savefig('fig_regret.pdf', bbox_inches='tight')
```

**输出：** Fig. 3

**关键观测点：**
- UCB-SpecStop vs Naive UCB1的收敛速度差异
- 大gap vs 小gap下的识别轮数
- regret曲线是否符合$O(\sqrt{t\log t})$

### 实验S4：Value of Information

**目标：** 验证Theorem 6 [1]

```python
def compute_voi(alpha, cd, cv, d_good, d_bad, 
                p_good, p_bad=None):
    """计算VOI"""
    if p_bad is None:
        p_bad = 1 - p_good
    
    # 稳态概率
    pi_good = p_bad / (p_good + p_bad)  
    # 注：这里p是离开当前状态的概率
    pi_bad = p_good / (p_good + p_bad)
    
    mu_D = pi_good * d_good + pi_bad * d_bad
    
    # Blind策略
    k_blind = compute_kstar(alpha, cd, cv, mu_D)
    c_blind = C(k_blind, mu_D, alpha, cd, cv)
    
    # State-dependent策略
    k_good = compute_kstar(alpha, cd, cv, d_good)
    k_bad = compute_kstar(alpha, cd, cv, d_bad)
    c_adaptive = (pi_good * C(k_good, d_good, alpha, cd, cv) 
                  + pi_bad * C(k_bad, d_bad, alpha, cd, cv))
    
    voi = c_blind - c_adaptive
    return voi, k_blind, k_good, k_bad

# 扫描bad-state mean
alpha = 0.7
cd, cv = 1.0, 0.5
d_good = 5.0  # ms
d_bad_range = np.arange(5, 500, 5)
p_transition = 0.1

vois = []
dc = dc_theory(alpha, cd, cv)

for d_bad in d_bad_range:
    voi, _, _, _ = compute_voi(alpha, cd, cv, d_good, d_bad, 
                                p_transition)
    vois.append(voi)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(d_bad_range, vois, 'b-', linewidth=2)
ax.axvline(x=dc, linestyle='--', color='red', 
           label=f'$d_c$ = {dc:.1f} ms')
ax.set_xlabel('Bad-state mean delay (ms)')
ax.set_ylabel('VOI (ms/token)')
ax.legend()
ax.set_title('Value of Information vs. Bad-State Delay')
plt.savefig('fig_voi.pdf', bbox_inches='tight')
```

**输出：** Fig. 4

### 实验S5：不同延迟分布的鲁棒性

**目标：** 验证Theorem 3（mean-sufficiency）在不同分布下的表现 [1]

```python
# 三种分布：确定性、指数、对数正态
distributions = {
    'Deterministic': lambda d: d,
    'Exponential': lambda d: np.random.exponential(d),
    'Log-Normal': lambda d: np.random.lognormal(
        np.log(d) - 0.5 * np.log(1 + 1), 
        np.sqrt(np.log(1 + 1)))  # variance = mean
}

# 对每种分布，Monte Carlo估计C(k,d)
def mc_cost(k, d_mean, dist_func, alpha, cd, cv, 
            n_samples=100000):
    total_N = 0.0
    total_A = 0.0
    for _ in range(n_samples):
        D = dist_func(d_mean)
        N = k * (cd + cv) + 2 * D + cv
        # 模拟acceptance
        n_acc = 0
        for i in range(k):
            if np.random.random() < alpha:
                n_acc += 1
            else:
                break
        A = n_acc + 1
        total_N += N
        total_A += A
    return total_N / total_A
```

---

## 四、硬件验证实验

### 硬件拓扑

```
Jetson Orin Nano Super (8GB)        3090 Server (24GB)
┌─────────────────────┐            ┌─────────────────────┐
│  Draft Model         │◄──────────►│  Target Model        │
│  Qwen2.5-0.5B        │  以太网     │  Qwen2.5-7B          │
│  or LLaMA-3.2-1B     │  +tc netem │  or LLaMA-3.2-8B     │
└─────────────────────┘            └─────────────────────┘
       Edge (cd ≈ 15ms/token)            Cloud (cv ≈ 3ms/token)
```

### 模型选择方案

| 实验组 | Draft (Jetson) | Target (3090) | 预期$\alpha$ |
|--------|---------------|---------------|-------------|
| 组A | Qwen2.5-0.5B | Qwen2.5-7B | ~0.6-0.7 |
| 组B | LLaMA-3.2-1B | LLaMA-3.2-8B | ~0.7-0.8 |

### 实验H0：基础参数测量

```python
# 在Jetson上测量cd
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_draft = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16)

# 测量单token生成延迟
input_ids = tokenizer.encode("Hello world", 
                              return_tensors="pt")
latencies = []
for _ in range(1000):
    start = time.perf_counter()
    with torch.no_grad():
        output = model_draft.generate(
            input_ids, max_new_tokens=1, 
            do_sample=False)
    latencies.append((time.perf_counter() - start) * 1000)

cd_measured = np.median(latencies)
print(f"Measured cd = {cd_measured:.2f} ms")

# 在3090上类似测量cv
# cv通常远小于cd（3090算力强很多）
```

```python
# 测量acceptance rate α
def measure_alpha(draft_model, target_model, tokenizer, 
                  dataset, max_k=10):
    """统计不同position的acceptance rate"""
    position_accepts = {i: [] for i in range(max_k)}
    
    for prompt in dataset:
        input_ids = tokenizer.encode(prompt, 
                                      return_tensors="pt")
        # Draft生成k个token
        draft_tokens = draft_model.generate(
            input_ids, max_new_tokens=max_k)
        # Target验证
        with torch.no_grad():
            target_logits = target_model(draft_tokens).logits
        
        for pos in range(max_k):
            # 比较draft和target分布，rejection sampling
            accepted = check_acceptance(
                draft_tokens, target_logits, pos)
            position_accepts[pos].append(accepted)
    
    # 计算每个position的acceptance rate
    per_position_alpha = {pos: np.mean(accepts) 
                          for pos, accepts in 
                          position_accepts.items()}
    
    # 整体α（geometric fit）
    alpha_overall = np.mean(list(
        per_position_alpha.values()))
    
    return alpha_overall, per_position_alpha
```

**关键输出：**
- 实测$c_d$、$c_v$、$\alpha$的具体值
- **验证Assumption 1的合理性**：画per-position $\alpha$图，展示geometric assumption的近似程度

### 实验H1：网络延迟模拟

```bash
# 在Jetson上配置tc netem模拟不同延迟
# 确定性延迟
sudo tc qdisc add dev eth0 root netem delay 10ms

# 指数分布延迟 (均值50ms, 抖动30ms)
sudo tc qdisc change dev eth0 root netem \
    delay 50ms 30ms distribution normal

# 重尾延迟 (pareto分布)
sudo tc qdisc change dev eth0 root netem \
    delay 100ms 50ms distribution pareto

# 扫描脚本
for delay in 5 10 30 50 100 200; do
    sudo tc qdisc change dev eth0 root netem \
        delay ${delay}ms
    python run_experiment.py --delay $delay \
        --output results_d${delay}.json
done
```

### 实验H2：端到端系统实现

```python
# edge_client.py (运行在Jetson上)
import socket
import json
import time

class EdgeDraftClient:
    def __init__(self, draft_model, server_addr):
        self.model = draft_model
        self.server = server_addr
    
    def speculate_and_verify(self, prompt, k):
        """完整的一轮speculative decoding"""
        # Step 1: Draft生成k个token
        t_start = time.perf_counter()
        draft_tokens = self.generate_draft(prompt, k)
        t_draft = time.perf_counter() - t_start
        
        # Step 2: 发送到云端验证
        t_comm_start = time.perf_counter()
        response = self.send_to_cloud(draft_tokens)
        t_comm = time.perf_counter() - t_comm_start
        
        # Step 3: 解析验证结果
        n_accepted = response['n_accepted']
        verified_tokens = response['tokens']
        t_verify = response['verify_time']
        
        # 计算指标
        N_t = t_draft + t_comm  # 总时间
        A_t = n_accepted + 1     # 含bonus token
        
        return {
            'k': k, 'N_t': N_t * 1000, 'A_t': A_t,
            't_draft': t_draft * 1000,
            't_comm': t_comm * 1000,
            't_verify': t_verify,
            'tokens': verified_tokens
        }

# cloud_server.py (运行在3090上)
class CloudVerifyServer:
    def __init__(self, target_model):
        self.model = target_model
    
    def verify(self, draft_tokens, context):
        """验证draft tokens并返回结果"""
        t_start = time.perf_counter()
        # 并行验证所有k个token
        logits = self.model(
            torch.cat([context, draft_tokens]))
        
        # Rejection sampling
        n_accepted = 0
        for i in range(len(draft_tokens)):
            if self.accept(draft_tokens[i], logits[i]):
                n_accepted += 1
            else:
                break
        
        # 生成bonus token
        bonus = self.model.generate(max_new_tokens=1)
        t_verify = (time.perf_counter() - t_start) * 1000
        
        return {
            'n_accepted': n_accepted,
            'tokens': accepted_tokens + [bonus],
            'verify_time': t_verify
        }
```

### 实验H3：全策略对比（核心硬件实验）

```python
def run_hardware_experiment(client, prompts, delay_ms, 
                            alpha_est, cd_est, cv_est):
    """对每个delay运行所有策略"""
    results = {}
    
    # B1: Fixed-k baselines
    for k in [1, 3, 5, 7, 10]:
        metrics = []
        for prompt in prompts:
            m = client.speculate_and_verify(prompt, k)
            metrics.append(m)
        results[f'fixed_k={k}'] = aggregate(metrics)
    
    # B2: Greedy (d=0 optimal)
    k_greedy = compute_kstar(alpha_est, cd_est, cv_est, d=0)
    metrics = []
    for prompt in prompts:
        m = client.speculate_and_verify(prompt, k_greedy)
        metrics.append(m)
    results['greedy'] = aggregate(metrics)
    
    # B4: SpecDec++ threshold
    k_spec = specdec_pp_policy(alpha_est, p_min=0.3)
    metrics = []
    for prompt in prompts:
        m = client.speculate_and_verify(prompt, k_spec)
        metrics.append(m)
    results['specdec++'] = aggregate(metrics)
    
    # B5: Oracle optimal
    k_oracle = compute_kstar(alpha_est, cd_est, cv_est, 
                              delay_ms)
    metrics = []
    for prompt in prompts:
        m = client.speculate_and_verify(prompt, k_oracle)
        metrics.append(m)
    results['oracle'] = aggregate(metrics)
    
    # Ours: UCB-SpecStop (online learning)
    ucb = UCBSpecStop(K_max=15, beta=1.0)
    metrics = []
    for prompt in prompts:
        k = ucb.select_arm()
        m = client.speculate_and_verify(prompt, k)
        ucb.update(k, m['N_t'], m['A_t'])
        metrics.append(m)
    results['ucb_specstop'] = aggregate(metrics)
    
    return results
```

### 实验H4：Markov信道下的自适应

```bash
# 模拟Markov信道：交替切换good/bad状态
# markov_channel.sh
STATE="good"
while true; do
    if [ "$STATE" = "good" ]; then
        sudo tc qdisc change dev eth0 root netem delay 5ms
        # 以概率0.1转移到bad
        if [ $(( RANDOM % 10 )) -eq 0 ]; then
            STATE="bad"
        fi
    else
        sudo tc qdisc change dev eth0 root netem delay 80ms
        if [ $(( RANDOM % 10 )) -eq 0 ]; then
            STATE="good"
        fi
    fi
    sleep 0.1  # 每100ms检查一次状态
done
```

### 实验H5：WiFi真实网络

```python
# 将Jetson和3090通过WiFi连接
# 不使用tc netem，让真实网络提供延迟和抖动
# 同时记录RTT作为网络状态
def measure_rtt(server_addr):
    start = time.perf_counter()
    sock.sendto(b'ping', server_addr)
    sock.recvfrom(1024)
    return (time.perf_counter() - start) * 1000 / 2
```

---

## 五、实验结果呈现

按照写作指南 [2]，每组实验应有Motivate → Observe → Interpret的结构：

### Section VI 重写框架

```latex
\section{Experimental Validation}
\label{sec:experiments}

We validate the theoretical predictions through Monte Carlo 
simulation (\S\ref{subsec:sim_setup}--\S\ref{subsec:sim_voi}) 
and hardware experiments on a Jetson Orin Nano Super (edge) 
with an NVIDIA RTX 3090 server (cloud) 
(\S\ref{subsec:hw_setup}--\S\ref{subsec:hw_markov}).

\subsection{Simulation Setup}
% ... 参数说明 ...

\subsection{Phase Transition Verification}
% Fig. 2: k*(d) curves
% 观测1: dc误差<X%，对数包络吻合

\subsection{Latency Improvement}
% Table I: 全baseline对比
% 观测2: optimal k*在d=100ms时比fixed k=5快Y%
% 观测3: Greedy (B2)在高延迟下性能急剧下降

\subsection{Online Learning Convergence}
% Fig. 3: regret curves
% 观测4: UCB-SpecStop vs Naive UCB1
% 观测5: 大gap下~Z轮识别最优arm

\subsection{Value of Information}
% Fig. 4: VOI curve
% 观测6: VOI在states跨越dc时达峰值

\subsection{Hardware Setup}
% Jetson + 3090拓扑
% 实测cd, cv, alpha

\subsection{Hardware Latency Validation}
% Table II: 硬件上的latency对比
% 观测7: 理论预测vs实测偏差<W%

\subsection{Adaptive Strategy under Markov Channel}
% 观测8: contextual UCB在Markov信道下
%   优于non-contextual X%
```

### 建议的图表数量

| 图/表 | 内容 | 来源 |
|-------|------|------|
| Fig. 1 | 系统架构图 | 已完成 |
| Fig. 2 | $k^*(d)$ phase transition | 仿真S1 |
| Fig. 3(a) | Regret (large gap) | 仿真S3 |
| Fig. 3(b) | Regret (small gap) | 仿真S3 |
| Fig. 4 | VOI curve | 仿真S4 |
| Fig. 5 | Per-position $\alpha$ validation | 硬件H0 |
| Table I | Simulation latency对比 | 仿真S2 |
| Table II | Hardware latency对比 | 硬件H3 |

INFOCOM限制9+1页 [2]，8张图表合理，如果超出可将Fig. 5放入supplementary material。

---

## 六、时间线建议

```
Week 1-2: 仿真实验S1-S5，填充论文所有TODO
Week 3:   购买Jetson + 搭建环境 + 部署模型
Week 4:   硬件实验H0 (参数测量) + H1 (网络模拟)
Week 5:   硬件实验H2-H3 (端到端对比)
Week 6:   硬件实验H4-H5 (Markov信道+WiFi)
Week 7:   整理数据 + 修改论文 + 内部审稿
Week 8:   投稿
```