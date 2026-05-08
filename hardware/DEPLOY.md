# UCB-SpecStop 硬件部署与实跑指南（按当前真实环境）

本文档基于当前实际情况整理，目标是让 Jetson + 3090 实验可直接跑通，并避免已出现的常见问题（Hugging Face 超时、路径不一致、Llama gated、netem 不可用）。

---

## 1. 实验架构与当前约束

- 边缘端（Jetson）运行 draft 小模型与实验主控脚本。
- 云端（3090）运行 `hardware/cloud_server.py` 提供 `/verify` 与 `/ping`。
- 当前环境中 `sch_netem` 可能不可用，实验主流程已支持 software-injected delay。
- 模型加载推荐本地路径，默认不依赖在线下载。

---

## 2. 代码与关键脚本

### Jetson 侧
- `hardware/measure_params.py`：H0，测 `cd/cv/alpha`。
- `hardware/run_revised_experiments.py`：R1-R6 主实验（推荐）。
- `hardware/run_all_suites.sh`：批量跑 qwen/llama/phi 三组。

### 云端 3090 侧
- `hardware/cloud_server.py`：验证服务。

---

## 3. 模型组（当前使用）

`run_revised_experiments.py` 内置 suite：

- `qwen`
  - draft: `Qwen/Qwen2.5-0.5B`
  - cloud: `Qwen/Qwen2.5-7B-Instruct`
- `llama`
  - draft: `meta-llama/Llama-3.2-1B-Instruct`
  - cloud: `meta-llama/Llama-3.1-8B-Instruct`
- `phi`
  - draft: `microsoft/Phi-3-mini-4k-instruct`
  - cloud: `microsoft/Phi-3-small-128k-instruct`

注意：Llama 模型通常是 gated，必须有授权账号/token。

---

## 4. 本地模型目录规范（强烈建议）

### Jetson
- `/home/jetson/local/models/Qwen2.5-0.5B`
- `/home/jetson/local/models/Llama-3.2-1B-Instruct`
- `/home/jetson/local/models/Phi-3-mini-4k-instruct`

### 3090
- `/home/<user>/local/models/Qwen2.5-7B-Instruct`
- `/home/<user>/local/models/Llama-3.1-8B-Instruct`
- `/home/<user>/local/models/Phi-3-small-128k-instruct`

不要在 Jetson 上使用 `/home/skk/...` 这种别的机器路径。

---

## 5. 下载模型（镜像）

可用镜像：`https://hf-mirror.com`

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir /home/jetson/local/models/Qwen2.5-0.5B

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir /home/jetson/local/models/Phi-3-mini-4k-instruct
```

Llama 如报 403（gated），先登录并确保账号已获批：

```bash
huggingface-cli login
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /home/jetson/local/models/Llama-3.2-1B-Instruct
```

---

## 6. 云端启动命令（3090）

每次只启动一组对应大模型（同端口 8000）：

```bash
# Qwen
python hardware/cloud_server.py \
  --model /home/<user>/local/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000

# Llama
python hardware/cloud_server.py \
  --model /home/<user>/local/models/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000

# Phi
python hardware/cloud_server.py \
  --model /home/<user>/local/models/Phi-3-small-128k-instruct \
  --host 0.0.0.0 --port 8000
```

如需在线下载（不推荐）才加 `--allow-download`。

---

## 7. Jetson 先做 H0 参数测量

每组模型都要单独测一份参数文件。

```bash
# Qwen
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Qwen2.5-0.5B \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_qwen.json

# Llama
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Llama-3.2-1B-Instruct \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_llama.json

# Phi
python hardware/measure_params.py \
  --draft-model /home/jetson/local/models/Phi-3-mini-4k-instruct \
  --server http://192.168.3.72:8000 \
  --prompts hardware/prompts.txt
mv outputs/hardware/params_measured.json outputs/hardware/params_phi.json
```

---

## 8. 跑修订版主实验（R1-R6）

### 单组运行

```bash
python hardware/run_revised_experiments.py \
  --suite qwen \
  --server http://192.168.3.72:8000 \
  --params outputs/hardware/params_qwen.json \
  --prompts hardware/prompts.txt \
  --n-prompts 500 \
  --n-rounds 200 \
  --exp all
```

输出目录：`outputs/hardware_revised/qwen/`

同理把 `suite/params` 改成 `llama`、`phi`。

### 批量运行

```bash
chmod +x hardware/run_all_suites.sh
SERVER=http://192.168.3.72:8000 N_PROMPTS=500 N_ROUNDS=200 EXP=all ./hardware/run_all_suites.sh
```

脚本会逐组运行，输出在 `outputs/hardware_revised/<suite>/`。

---

## 9. 当前关键差异（相对于旧文档）

1. 主流程以 `run_revised_experiments.py` 为准，不再优先旧 `run_hw_experiment.py`。
2. 真实环境下 `sch_netem` 不可用时，使用 software-injected delay（已在脚本内实现）。
3. 模型路径优先本地目录，避免 HF 网络超时。
4. Llama 需 gated 授权，镜像也无法绕过权限。
5. 实验日志字段已经统一为：
   - `configured_one_way_delay_ms`
   - `bare_rtt_ms`
   - `measured_comm_round_ms`
   - `accepted_draft_len`
   - `accepted_total`
   - `total_round_time_ms`

---

## 10. 常见问题与处理

### Q1: `Incorrect path_or_model_id` / `HFValidationError`
原因：传了错误机器路径（如 Jetson 上使用 `/home/skk/...`）。
处理：改成 Jetson 本机路径（`/home/jetson/local/models/...`）。

### Q2: `Connection to huggingface.co timed out`
原因：外网不稳定。
处理：
- 首选本地模型路径。
- 需要下载时用 `HF_ENDPOINT=https://hf-mirror.com`。

### Q3: Llama 下载 403
原因：gated 模型未授权。
处理：`huggingface-cli login` 并确认账号已获批。

### Q4: `attention mask is not set` warning
不是致命错误，流程可继续；建议使用仓库最新脚本版本。

### Q5: 如何确认云端在线
Jetson 执行：
```bash
curl http://192.168.3.72:8000/ping
```
返回 `{"status":"ok",...}` 即正常。

---

## 11. 结果文件检查

```bash
find outputs/hardware_revised -type f \( \
  -name "r1_calibration.csv" -o \
  -name "r2_acceptance.csv" -o \
  -name "r3_phase_transition.csv" -o \
  -name "r4_strategy_compare.csv" -o \
  -name "r5_regret_data.npz" -o \
  -name "r6_markov_voi.json" -o \
  -name "run_config.json" \
\) | sort
```

---

## 12. 推荐执行顺序

1. 云端启动对应大模型。
2. Jetson 跑对应 draft 的 `measure_params.py` 并保存 `params_<suite>.json`。
3. Jetson 跑 `run_revised_experiments.py --suite <suite> --params ...`。
4. 三组都跑完后统一对比 `outputs/hardware_revised/*`。
