# UCB-SpecStop 硬件实验部署指南

## 系统架构

```
┌─────────────────────────────────────┐      以太网 (tc netem)      ┌──────────────────────────────────┐
│   边缘端 — Jetson Orin Nano Super   │ ◄──────────────────────────► │   云端 — RTX 3090 服务器         │
│   (8 GB RAM, Ampere GPU)            │                              │   (24 GB VRAM)                   │
│                                     │                              │                                  │
│  Draft Model (小模型):              │  POST /verify                │  Target Model (大模型):           │
│    Qwen2.5-0.5B  (组A)             │  ──────────────────────────► │    Qwen2.5-7B   (组A)            │
│    LLaMA-3.2-1B  (组B)             │                              │    LLaMA-3.2-8B (组B)            │
│                                     │  GET /ping (RTT测量)         │                                  │
│  运行文件:                          │ ◄────────────────────────── │  运行文件:                        │
│    edge_client.py                   │                              │    cloud_server.py               │
│    run_hw_experiment.py             │                              │                                  │
│    measure_params.py                │                              │                                  │
│    markov_netem.py (H4实验)         │                              │                                  │
│    setup_netem.sh  (本地netem)      │                              │                                  │
└─────────────────────────────────────┘                              └──────────────────────────────────┘
```

---

## 文件分布总览

### 边缘端 (Jetson) 运行的文件

| 文件 | 用途 | 何时使用 |
|------|------|----------|
| `hardware/edge_client.py` | 主客户端：生成draft tokens，发送到云端验证，记录结果 | H1–H5 所有实验 |
| `hardware/run_hw_experiment.py` | 实验主控脚本：H1–H4 全流程编排 | H1–H4 |
| `hardware/measure_params.py` | H0：测量 cd、cv、alpha 基础参数 | 最先运行，一次性 |
| `hardware/markov_netem.py` | H4：Markov 信道切换守护进程（需 sudo） | H4/E5 实验 |
| `hardware/setup_netem.sh` | 手动配置 tc netem 延迟 | 调试/手动操作 |
| `hardware/validate_hw.py` | 离线校验：15项单元测试，不需要真实硬件 | 部署前必跑 |
| `src/` (整个目录) | 核心算法库（UCBSpecStop、baselines、core等） | 所有脚本依赖 |
| `hardware/requirements_hw.txt` | Python 依赖 | 环境安装时 |

### 云端 (3090) 运行的文件

| 文件 | 用途 | 何时使用 |
|------|------|----------|
| `hardware/cloud_server.py` | FastAPI 服务端：加载 target model，执行 rejection sampling 验证 | H0–H5 全程保持运行 |
| `hardware/requirements_hw.txt` | Python 依赖 | 环境安装时 |
| `src/` (整个目录) | 核心算法库 | cloud_server 依赖 |

> **注意**：`setup_netem.sh` 和 `markov_netem.py` 在 **Jetson 本地网口** 上配置 netem（推荐），
> 也可以通过 `--server-user` SSH 到 3090 服务器配置（若需要控制入站流量方向）。

---

## 第一步：准备工作（两端都做）

### 1.1 确认硬件和系统

**Jetson 端：**
```bash
# 确认 JetPack 版本 (需 >= 6.0)
jetson_release
# 期望输出类似: JetPack 6.0 [L4T 36.x.x]

# 确认 CUDA 可用
nvidia-smi
# 期望看到: Orin (8GB) GPU

# 确认 Python 版本 (需 >= 3.10)
python3 --version
```

**3090 服务器端：**
```bash
# 确认 CUDA 12.x
nvidia-smi
python3 --version   # 需 >= 3.10
```

### 1.2 网络连通性确认

```bash
# 在 Jetson 上 ping 3090（记录这个 IP，后续称为 <3090-IP>）
ping <3090-IP>

# 确认 8000 端口可以访问（先不用 server 跑，只验证网络）
nc -zv <3090-IP> 8000   # 暂时会显示 refused，正常
```

### 1.3 克隆代码（两端都需要完整代码库）

```bash
# 两端都执行:
git clone <your-repo-url> ~/UCB-SpecStop
cd ~/UCB-SpecStop
```

---

## 第二步：Jetson 端环境配置

### 2.1 安装 PyTorch（Jetson 专用 wheel，不能用 pip install torch）

```bash
# 方法 A：官方 NVIDIA PyPI（推荐）
pip install torch torchvision --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v61

# 方法 B：如果 A 失败，从 Jetson 论坛下载对应 JetPack wheel
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# 示例（JetPack 6.0, Python 3.10）：
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+6a974be-cp310-cp310-linux_aarch64.whl
pip install torch-2.2.0a0+6a974be-cp310-cp310-linux_aarch64.whl
```

### 2.2 安装其余依赖

```bash
cd ~/UCB-SpecStop
pip install -r hardware/requirements_hw.txt
# 注意：transformers, accelerate, fastapi, uvicorn, requests, numpy,
#        pandas, matplotlib, scipy 均在此安装
```

### 2.3 验证 GPU 在 Jetson 上可用

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
# 期望: True  Orin (或类似)
```

### 2.4 运行离线校验（必须 15/15 通过才继续）

```bash
cd ~/UCB-SpecStop
python hardware/validate_hw.py
# 期望输出:
#   PASS  test_B_formula
#   PASS  test_C_formula
#   ... (15 项)
#   15 passed, 0 failed
```

---

## 第三步：3090 服务器端环境配置

### 3.1 安装依赖

```bash
cd ~/UCB-SpecStop
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r hardware/requirements_hw.txt
```

### 3.2 下载目标模型（3090 端，第一次运行时自动下载，建议提前缓存）

```bash
# 预下载（可选，避免实验时等待）
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', torch_dtype='auto')"
```

### 3.3 启动云端服务（3090 端，全程保持此窗口运行）

```bash
cd ~/UCB-SpecStop

# 组A: Qwen 模型对
python hardware/cloud_server.py \
    --model Qwen/Qwen2.5-7B \
    --host 0.0.0.0 \
    --port 8000

# 或 组B: LLaMA 模型对
# python hardware/cloud_server.py \
#     --model meta-llama/Llama-3.2-8B \
#     --host 0.0.0.0 \
#     --port 8000

# 期望看到: "Target model loaded." 和 Uvicorn 启动日志
```

### 3.4 验证服务器响应（从 Jetson 测试）

```bash
# 在 Jetson 上
curl http://<3090-IP>:8000/ping
# 期望: {"status":"ok","ts":...}
```

---

## 第四步：H0 基础参数测量（Jetson 端）

**必须第一个运行。** 测量 cd（草稿延迟）、cv（验证延迟）、alpha（接受率），
输出 `outputs/hardware/params_measured.json`，后续所有实验依赖此文件。

### 4.1 准备 prompts 文件

```bash
# 创建测试 prompt 文件（至少 200 行）
# 方法 A：使用 MT-Bench 样例（推荐）
cat > ~/UCB-SpecStop/hardware/prompts.txt << 'EOF'
Write a short story about a robot learning to paint.
Explain quantum entanglement in simple terms.
What are the main differences between Python and JavaScript?
Describe the water cycle in detail.
How does machine learning differ from traditional programming?
What causes the seasons to change throughout the year?
Summarize the key events of World War II.
Explain how neural networks process information.
What are the health benefits of regular exercise?
Describe how a compiler works step by step.
EOF
# 实际实验建议用 200+ 条 ShareGPT/MT-Bench prompts

# 方法 B：从 HuggingFace 下载 (需网络)
python3 -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/mt_bench_prompts', split='train')
with open('hardware/prompts.txt', 'w') as f:
    for row in ds['prompt'][:200]:
        f.write(row.replace('\n', ' ') + '\n')
print('写入', len(open('hardware/prompts.txt').readlines()), '条 prompts')
"
```

### 4.2 运行 H0 参数测量

```bash
cd ~/UCB-SpecStop

# 在 Jetson 上运行（3090 服务器必须已启动）
python hardware/measure_params.py \
    --draft-model Qwen/Qwen2.5-0.5B \
    --server http://<3090-IP>:8000 \
    --prompts hardware/prompts.txt \
    --k-max 10

# 期望输出示例：
#   cd = 14.73 ms/token    (Jetson 生成1个token的延迟)
#   cv = 2.81 ms/token     (3090 验证1个token的延迟)
#   alpha (mean) = 0.682
#   pos 0: 0.814  (geometric: 0.682)
#   pos 1: 0.711  (geometric: 0.465)
#   ...
#   Saved to outputs/hardware/params_measured.json
#   Saved Fig. 7 (alpha validation) to outputs/hardware/fig_alpha_per_pos.pdf
```

### 4.3 确认输出文件

```bash
cat outputs/hardware/params_measured.json
# 应包含:
# {
#   "cd_ms": 14.73,
#   "cv_ms": 2.81,
#   "alpha_fit": 0.682,
#   "per_position_alpha": {"0": 0.814, "1": 0.711, ...}
# }
```

---

## 第五步：配置 tc netem（Jetson 端，需 sudo）

### 5.1 初始化 netem（只需执行一次）

```bash
# 查看网络接口名称
ip link show
# 通常是 eth0 或 enp3s0，用实际名称替换下面的 eth0

# 添加 netem 规则（初始设为 50ms）
sudo bash ~/UCB-SpecStop/hardware/setup_netem.sh add eth0 50
# 期望: "Added netem: delay=50ms jitter=0ms dist=deterministic"
```

### 5.2 验证延迟生效

```bash
ping <3090-IP>
# 每次 RTT 应接近 100ms（50ms 单程 × 2）
```

### 5.3 常用 netem 命令参考

```bash
# 变更延迟（实验过程中用 change，不用 add）
sudo bash hardware/setup_netem.sh change eth0 10      # 10ms 确定性
sudo bash hardware/setup_netem.sh change eth0 50      # 50ms 确定性
sudo bash hardware/setup_netem.sh change eth0 100     # 100ms 确定性
sudo bash hardware/setup_netem.sh change eth0 50 30 normal    # 正态抖动
sudo bash hardware/setup_netem.sh change eth0 100 50 pareto   # 重尾分布

# 删除 netem（恢复正常网络）
sudo bash hardware/setup_netem.sh del eth0
```

---

## 第六步：运行硬件实验 H1–H4（Jetson 端）

### 方式 A：一键运行全部实验（推荐）

```bash
cd ~/UCB-SpecStop

python hardware/run_hw_experiment.py --all \
    --draft-model Qwen/Qwen2.5-0.5B \
    --server http://<3090-IP>:8000 \
    --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt \
    --n-prompts 200 \
    --delays 5 10 30 50 100 \
    --n-regret-rounds 1000 \
    --n-markov-rounds 500 \
    --k-max 20 \
    --rejection-sampling

# 全部完成后，outputs/hardware/ 下会有:
#   h1_kstar_sweep.csv          — H1/E1 数据
#   fig_h1_kstar_sweep.pdf      — Fig: k* vs log(d)
#   h2_phase_transition.csv     — H2/E2 数据
#   fig_h2_phase_transition.pdf — Fig: phase transition
#   h3_strategy_compare.csv     — H3/E3+E4 数据
#   table_ii_hw_comparison.csv  — Table II
#   fig_h3_strategy_compare.pdf — Fig: bar chart
#   fig_h3_regret_curve.pdf     — Fig: cumulative regret
#   h3_regret_data.npz          — 原始 regret 数据
#   h4_markov_voi.json          — H4/E5 VOI 数值
#   fig_h4_markov_regret.pdf    — Fig: Markov regret
```

### 方式 B：逐个实验运行（便于调试）

```bash
# H1/E1: k* sweep (验证 log scaling)
python hardware/run_hw_experiment.py --exp h1 \
    --server http://<3090-IP>:8000 --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt \
    --delays 5 10 30 50 100 200

# H2/E2: Phase transition (验证相变)
python hardware/run_hw_experiment.py --exp h2 \
    --server http://<3090-IP>:8000 --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt

# H3/E3+E4: 全策略对比 + regret 曲线
python hardware/run_hw_experiment.py --exp h3 \
    --server http://<3090-IP>:8000 --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt --n-prompts 200 \
    --delays 5 10 30 80 \
    --n-regret-rounds 1000

# H4/E5: Markov 信道 VOI
python hardware/run_hw_experiment.py --exp h4 \
    --server http://<3090-IP>:8000 --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt \
    --d-good 5 --d-bad 80 \
    --n-markov-rounds 500
```

---

## 第七步：H4 Markov 信道实验——双终端操作

H4 实验需要同时控制 netem 切换，使用两个终端：

**Jetson 终端 1（netem 守护进程，必须 sudo）：**
```bash
sudo python ~/UCB-SpecStop/hardware/markov_netem.py \
    --iface eth0 \
    --d-good 5 \
    --d-bad 80 \
    --p-g2b 0.1 \
    --p-b2g 0.1 \
    --interval 0.1

# 此进程持续运行，每 0.1 秒按 Markov 概率切换 netem 延迟
# 当前状态写入 /tmp/markov_netem_state.txt
# Ctrl+C 停止并自动清理 netem 规则
```

**Jetson 终端 2（实验本体）：**
```bash
python ~/UCB-SpecStop/hardware/run_hw_experiment.py --exp h4 \
    --server http://<3090-IP>:8000 \
    --iface eth0 \
    --params outputs/hardware/params_measured.json \
    --prompts hardware/prompts.txt \
    --d-good 5 --d-bad 80 \
    --n-markov-rounds 500
```

---

## 第八步：单策略快速测试（edge_client.py）

`edge_client.py` 可以独立运行单个策略，适合快速验证某个 baseline：

```bash
cd ~/UCB-SpecStop

# 测试 UCB-SpecStop (我们的算法)
python hardware/edge_client.py \
    --draft-model Qwen/Qwen2.5-0.5B \
    --server http://<3090-IP>:8000 \
    --strategy ucb \
    --delay 50 \
    --prompts hardware/prompts.txt \
    --n-prompts 100 \
    --output outputs/hardware/test_ucb_d50.json \
    --rejection-sampling

# 测试 Fixed-k=5 baseline (B1)
python hardware/edge_client.py \
    --strategy fixed5 \
    --server http://<3090-IP>:8000 \
    --prompts hardware/prompts.txt \
    --output outputs/hardware/test_fixed5_d50.json

# 测试 Naive UCB (B6，用于对比)
python hardware/edge_client.py \
    --strategy naive_ucb \
    --server http://<3090-IP>:8000 \
    --prompts hardware/prompts.txt \
    --output outputs/hardware/test_naivucb_d50.json

# 仅测量 RTT（不跑实验）
python hardware/edge_client.py \
    --draft-model Qwen/Qwen2.5-0.5B \
    --server http://<3090-IP>:8000 \
    --measure-rtt

# 仅测量 cd（不需要服务器）
python hardware/edge_client.py \
    --draft-model Qwen/Qwen2.5-0.5B \
    --server http://<3090-IP>:8000 \
    --measure-cd
```

---

## 常见问题排查

### Q: Jetson 上 `import torch` 失败或 CUDA 不可用
```bash
# 检查是否装了 x86 版本的 torch（错误）
pip show torch | grep -i location
# 如果路径正常但 cuda 不可用，重新按 Step 2.1 方法 B 安装 aarch64 wheel
```

### Q: 连接 3090 服务器超时
```bash
# 检查防火墙
# 在 3090 上：
sudo ufw allow 8000/tcp
# 或直接关闭防火墙调试：
sudo ufw disable

# 确认服务器监听正确接口
ss -tlnp | grep 8000
# 应显示 0.0.0.0:8000
```

### Q: tc netem 报错 "RTNETLINK answers: File exists"
```bash
# 说明 netem 已存在，用 change 而不是 add
sudo bash hardware/setup_netem.sh change eth0 50
# 或先删除再添加：
sudo bash hardware/setup_netem.sh del eth0
sudo bash hardware/setup_netem.sh add eth0 50
```

### Q: `params_measured.json` 中 alpha 太低（< 0.4）
说明模型对 (0.5B, 7B) 对齐程度低，考虑：
- 换用 LLaMA-3.2 1B → 8B 对（同家族对齐更好）
- 或增大 prompt 数量（`--k-max 5` 而非 10，减少截断偏差）

### Q: 实验运行太慢（每轮超过 5 秒）
```bash
# 检查是否在 Jetson 上用了 CPU（应该用 GPU）
# 在 edge_client.py 启动时应看到: "Loading draft model ... device=cuda"

# 如果显示 device=cpu，说明 torch 没识别到 Jetson GPU，重新安装 wheel
```

### Q: `validate_hw.py` 中某项 FAIL
```bash
# 重新运行单项测试（找到失败的测试函数名后）
python3 -c "
import sys; sys.path.insert(0, '.')
from hardware.validate_hw import test_ucb_uses_ratio_of_sums
test_ucb_uses_ratio_of_sums()
print('OK')
"
```

---

## 实验产出文件列表

运行全部实验后，`outputs/hardware/` 目录结构：

```
outputs/hardware/
├── params_measured.json          # H0: cd, cv, alpha 参数
├── fig_alpha_per_pos.pdf         # H0: per-position alpha 图 (Fig.5)
│
├── h1_kstar_sweep.csv            # H1/E1: k* sweep 原始数据
├── fig_h1_kstar_sweep.pdf        # H1/E1: k* vs log(d) 图
│
├── h2_phase_transition.csv       # H2/E2: 相变扫描数据
├── fig_h2_phase_transition.pdf   # H2/E2: 相变图
│
├── h3_strategy_compare.csv       # H3/E3+E4: 全策略对比数据
├── table_ii_hw_comparison.csv    # Table II: 论文用表格
├── fig_h3_strategy_compare.pdf   # H3: 各策略 bar chart
├── fig_h3_regret_curve.pdf       # H3/E3: cumulative regret 曲线 (Fig.3)
├── h3_regret_data.npz            # H3: regret 原始 numpy 数据
│
├── h4_markov_voi.json            # H4/E5: VOI 数值结果
└── fig_h4_markov_regret.pdf      # H4/E5: Markov regret 对比图 (Fig.4)
```

---

## 参数速查表

| 参数 | 典型值（Qwen组A） | 说明 |
|------|----------|------|
| `alpha` | 0.65–0.75 | 接受率，H0 实测 |
| `cd` | 13–18 ms | Jetson 0.5B 生成1 token |
| `cv` | 2–4 ms | 3090 验证1 token |
| `d_c` | 约 2–4 ms | 理论相变延迟 |
| `k*`(d=10ms) | 2–4 | 理论最优 draft 长度 |
| `k*`(d=50ms) | 5–8 | 理论最优 draft 长度 |
| `k*`(d=100ms) | 7–12 | 理论最优 draft 长度 |
| `K_max` | 20 | 最大 arm 数 |
| `beta` | 1.0 | UCB 探索系数 |
| `T`(regret) | 1000 | E3 regret 实验轮数 |
