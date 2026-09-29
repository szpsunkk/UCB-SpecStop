# UCB-SpecStop 实验复现版

英文主说明见 [README.md](README.md)。本版独立整理，保留原仓库主分支和
`v1.0`；论文、回复信、SSH 配置和模型权重不在公开包中。

## 无 GPU 复算最新表 4

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/reproduce.py table4
```

输出在 `outputs/table4/`：

- `table4_compact.md`、`table4_compact.csv`：与新版论文一致的成本、相对 B7
  的降低比例和云端交互次数。
- `table4_all_metrics.csv`：包括质量、TTFT、E2E 和通信负担的完整统计。
- `session_costs.csv`、`latency_contrasts.json`：会话级原值、八项配对比较和
  Holm 校正结果。
- `quality.json`、`audit.json`：质量统计与执行/输出一致性审计。

数据为已采集的 576 个请求，8 个重启会话，32 个题目。离线复算不等于重新
测量硬件时延。所有算法共享的目标模型质量收益，也不是 SN 独有的模型能力提升。

## 其他入口

```bash
python scripts/reproduce.py exact-replay
python scripts/reproduce.py markov
python scripts/reproduce.py figures
python scripts/reproduce.py simulations --quick
bash scripts/smoke.sh
```

实机重跑按照 [HARDWARE.md](docs/HARDWARE.md) 设置自己的设备地址和模型目录。
新实验计划写到 `outputs/`，不会覆盖归档。原实验未记录精确模型权重 commit，
因此不能承诺跨设备、驱动和权重版本逐比特一致；运行时会检查输出一致性，
失败时停止并保留记录。

整理过程只修改了路径、配置、发布入口与核验方式，保留原选臂规则、数值、
样本和统计口径。公开版不包含此前所有探索性实验，完整覆盖范围见
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。
