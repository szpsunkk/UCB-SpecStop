"""Audit the completed campaign and build every capability/latency table from raw data."""
import csv
import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.table4.score import score

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/table4"
OUTPUT = ROOT / "outputs/table4"
LABELS = [
    ("edge_only", "Edge-only (0.5B)"),
    ("fixed_k5", "B1: fixed k=5"),
    ("greedy", "B2: greedy k=1"),
    ("specdec_pp_threshold", "B3: SpecDec++ adaptation"),
    ("b4_idealized", "B4: idealized geometric"),
    ("b5_calibrated_geometric", "B5: constant-cost geometric"),
    ("b6_calibrated_fixed", "B6: calibration-fixed"),
    ("per_round_ucb_warm", "Per-round UCB"),
    ("ucbspec_release_warm", "Throughput-UCBSpec (common bonus)"),
    ("ucbspec_arm_bonus_warm", "Throughput-UCBSpec (arm bonus)"),
    ("sn_warm", "UCB-SpecStop (SN)"),
    ("direct_target", "Target-only streaming (7B)"),
]
PATH_LABELS = {"lan_direct": "Local Wi-Fi", "lan_plus150": "Wi-Fi with delay proxy"}


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    avg = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values)-1) * stats.sem(values))
    return [avg, avg-half, avg+half]


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (len(values)-rank)))
        result[index] = running
    return result.tolist()


def paired_test(differences):
    d = np.asarray(differences, dtype=float)
    sd = float(d.std(ddof=1))
    p = float(stats.ttest_1samp(d, 0).pvalue) if sd else (1.0 if d.mean() == 0 else 0.0)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=len(d))))
    flip_p = float(np.mean(np.abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-12))
    return dict(mean_difference_ms_token=mean_ci(d)[0], ci_low=mean_ci(d)[1], ci_high=mean_ci(d)[2],
                paired_t_p=p, sign_flip_p=flip_p, session_differences=d.tolist())


def quality_summary(pairs, key):
    edge = np.array([int(pair["edge"][key]) for pair in pairs])
    target = np.array([int(pair["target"][key]) for pair in pairs])
    delta = target-edge
    rng = np.random.default_rng(2026092802)
    sampled = delta[rng.integers(0, len(delta), size=(50000, len(delta)))].mean(axis=1)*100
    gains, losses = int(np.sum(delta == 1)), int(np.sum(delta == -1))
    p = min(1.0, float(2*stats.binom.cdf(min(gains, losses), gains+losses, 0.5))) if gains+losses else 1.0
    return dict(n=len(pairs), edge_correct=int(edge.sum()), target_correct=int(target.sum()),
                edge_accuracy_pct=float(edge.mean()*100), target_accuracy_pct=float(target.mean()*100),
                delta_pp=float(delta.mean()*100), delta_ci_pp=np.quantile(sampled, [0.025, 0.975]).tolist(),
                improved_questions=gains, worsened_questions=losses, exact_mcnemar_p=p)


def key(row):
    return (row["measurement_session"], row["condition_id"], row["method"], row["prompt_id"])


def enrich(row, rounds):
    row = dict(row)
    speculative = row["method"] not in ("direct_target", "edge_only")
    row["application_kib"] = (row["request_payload_bytes"]+row["response_payload_bytes"])/1024
    row["verification_rounds"] = len(rounds)
    row["generation_interactions"] = len(rounds) if speculative else int(row["method"] == "direct_target")
    if speculative:
        row["verification_rpc_ms"] = sum(r["verify_rpc_ms"] for r in rounds)
        row["verification_server_ms"] = sum(r["cloud_verify_time_ms"] for r in rounds)
        row["verification_residual_ms"] = row["verification_rpc_ms"]-row["verification_server_ms"]
        row["edge_generation_wall_ms"] = row["edge_prefill_time_ms"]+sum(
            r["draft_time_ms"]+r["draft_commit_time_ms"] for r in rounds)
        row["target_stream_residual_ms"] = None
    else:
        row["verification_rpc_ms"] = row["verification_server_ms"] = row["verification_residual_ms"] = None
        row["target_stream_residual_ms"] = (row["completion_latency_ms"]-row["server_total_ms"]
                                             if row["method"] == "direct_target" else None)
        row["edge_generation_wall_ms"] = row["completion_latency_ms"] if row["method"] == "edge_only" else 0.0
    return row


def aggregate(rows):
    sessions = defaultdict(list)
    for row in rows:
        sessions[row["measurement_session"]].append(row)
    assert len(sessions) == 8 and all(len(r) == 4 for r in sessions.values())
    ratios = {s: sum(r["completion_latency_ms"] for r in rs)/sum(r["generated_tokens"] for r in rs)
              for s, rs in sorted(sessions.items())}
    cost = mean_ci(list(ratios.values()))
    result = dict(requests=len(rows), sessions=len(sessions), cost_ms_token=cost[0],
                  cost_ci_low=cost[1], cost_ci_high=cost[2],
                  e2e_s=float(np.mean([r["completion_latency_ms"] for r in rows])/1000),
                  e2e_p95_s=float(np.quantile([r["completion_latency_ms"] for r in rows], .95)/1000),
                  ttft_ms=float(np.mean([r["ttft_ms"] for r in rows if r["ttft_ms"] is not None])),
                  ttft_observed_requests=sum(r["ttft_ms"] is not None for r in rows),
                  output_tokens=float(np.mean([r["generated_tokens"] for r in rows])),
                  cap_requests=sum(r["stop_reason"] == "cap" for r in rows),
                  answer_requests=sum(r["stop_reason"] == "answer" for r in rows),
                  eos_requests=sum(r["stop_reason"] == "eos" for r in rows))
    for field in ("application_kib", "generation_interactions", "verification_rounds", "verification_rpc_ms",
                  "verification_server_ms", "verification_residual_ms", "edge_generation_wall_ms",
                  "target_stream_residual_ms", "controller_time_ms", "cleanup_time_ms"):
        values = [r[field] for r in rows]
        result[field] = float(np.mean(values)) if all(v is not None for v in values) else None
    return result, ratios


def audit():
    campaign = read_json(WORK / "campaign.json")
    assert campaign["completed"] and len(campaign["sessions"]) == 8
    frozen = read_json(WORK / "frozen_manifest.json")
    for name, expected in frozen["files"].items():
        assert digest(WORK / name) == expected, name
    assert digest(Path(__file__).with_name("score.py")) == frozen["scoring_sha256"]
    protocol = read_json(WORK / "protocol/protocol.json")
    gold = read_json(WORK / "gold_answers.json")
    source = read_rows(WORK / "gsm8k_test.jsonl")
    for pid, item in gold.items():
        assert pid == f"gsm8k-test-{item['source_index']}"
        assert source[item["source_index"]]["answer"].rsplit("####", 1)[1].strip() == item["answer"]
    all_rows, all_rounds, equality = [], [], []
    for plan_path in sorted((WORK / "protocol").glob("capability-*-plan.json")):
        plan = read_json(plan_path)
        folder = WORK / "hardware_run" / plan["measurement_session"]
        manifest = read_json(folder / "run_manifest.json")
        assert manifest["completed"] and manifest["output_mismatches"] == manifest["failure_rows"] == 0
        assert set(manifest["target_native_eos_ids"]) == {151643, 151645}
        assert manifest["plan_sha256"] == digest(plan_path)
        assert read_json(folder / "frozen_plan.json") == plan
        prompts = read_rows(WORK / "protocol" / plan["prompts_file"])
        expected = {(plan["measurement_session"], "edge_only", "edge_only", p["prompt_id"]) for p in prompts}
        expected |= {(plan["measurement_session"], c["condition_id"], m, p["prompt_id"])
                     for c in plan["conditions"] for m in c["methods"] for p in prompts}
        rows = read_rows(folder / "request_metrics.jsonl")
        assert len(rows) == len(expected) == manifest["request_count"] == 72
        assert {key(row) for row in rows} == expected
        for row in rows:
            assert row["generated_tokens"] == len(json.loads(row["delivered_token_ids_json"])) >= 0
            eos_ids = manifest["native_edge_eos"] if row["method"] == "edge_only" else manifest["target_native_eos_ids"]
            assert not set(eos_ids) & set(json.loads(row["delivered_token_ids_json"]))
            assert row["stop_reason"] in ("cap", "eos", "answer")
        checks = read_rows(folder / "output_equality.jsonl")
        assert len(checks) == 4 and all(c["equal"] and c["checked_methods"] == 17 for c in checks)
        equality.extend(checks)
        all_rows.extend(rows)
        all_rounds.extend(read_rows(folder / "round_metrics.jsonl"))
    assert len(all_rows) == 576 and {r["prompt_id"] for r in all_rows} == set(gold)
    round_groups = defaultdict(list)
    for row in all_rounds:
        round_groups[key(row)].append(row)
    for row in all_rows:
        if row["method"] not in ("edge_only", "direct_target"):
            assert row["target_session_closed"] is True
            rounds = round_groups[key(row)]
            assert len(rounds) == row["rounds"]
            assert sum(r["delivered_tokens"] for r in rounds) == row["generated_tokens"]
            if row["method"].startswith("fixed_depth_"):
                assert all(r["selected_arm"] == int(row["method"].rsplit("_", 1)[1]) for r in rounds)
            if row["stop_reason"] == "answer":
                assert rounds[-1]["arm_censored"] and not rounds[-1]["feedback_updated"]
    by_prompt = defaultdict(list)
    for row in all_rows:
        by_prompt[row["prompt_id"]].append(row)
    for pid, rows in by_prompt.items():
        exact = [r for r in rows if r["method"] != "edge_only"]
        assert len({(r["delivered_token_ids_json"], r["stop_reason"]) for r in exact}) == 1, pid
    report = dict(passed=True, requests=576, questions=32, restarted_sessions=8,
                  exact_output_checks=32, exact_requests=544, failures=0, public_archive_hashes_verified=True,
                  frozen_scoring_verified=True, final_answer_round_censoring_verified=True,
                  gold_answers_verified_against_source=True, native_eos_exclusion_verified=True,
                  target_session_cleanup_verified=True,
                  fixed_depth_alias_execution_verified=True,
                  original_paper_modified=False, analysis_sha256=digest(Path(__file__)))
    return protocol, gold, [enrich(r, round_groups[key(r)]) for r in all_rows], report


def main():
    global WORK, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=WORK)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    WORK, OUTPUT = args.data_dir.resolve(), args.out_dir.resolve()
    if OUTPUT == WORK or WORK in OUTPUT.parents:
        parser.error("write generated results outside the archived data directory")
    if WORK == ROOT / "data/table4":
        from scripts.verify_data import verify
        verify()
    protocol, gold, rows, audit_result = audit()
    result_dir = OUTPUT
    result_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    by_prompt = defaultdict(dict)
    for row in rows:
        groups[(row["condition_id"], row["method"])].append(row)
        by_prompt[row["prompt_id"]][(row["condition_id"], row["method"])] = row
    pairs, scored = [], []
    for pid, methods in sorted(by_prompt.items()):
        e = methods[("edge_only", "edge_only")]
        t = methods[("lan_direct", "direct_target")]
        edge, target = score(e["output_text"], gold[pid]["answer"]), score(t["output_text"], gold[pid]["answer"])
        pairs.append(dict(prompt_id=pid, edge=edge, target=target))
        for label, row, scoring in (("edge", e, edge), ("target_shared", t, target)):
            scored.append(dict(prompt_id=pid, method=label, **scoring, stop_reason=row["stop_reason"],
                               tokens=row["generated_tokens"], output_text=row["output_text"]))
    quality = {name: quality_summary(pairs, key_name) for name, key_name in
               (("primary", "flexible_correct"), ("strict", "strict_correct"))}
    summaries, ratios = {}, {}
    for identity, method_rows in groups.items():
        summaries[identity], ratios[identity] = aggregate(method_rows)
    table, contrast_rows = [], []
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        for method, label in LABELS:
            actual = ("fixed_depth_%02d" % condition["aliases"][method]) if method in condition["aliases"] else method
            identity = ("edge_only", "edge_only") if method == "edge_only" else (cid, actual)
            summary = summaries[identity]
            qprefix = "edge" if method == "edge_only" else "target"
            edge_cost = summaries[("edge_only", "edge_only")]["cost_ms_token"]
            table.append(dict(condition=cid, method=method, label=label, actual_execution=actual,
                accuracy_pct=quality["primary"][qprefix+"_accuracy_pct"],
                accuracy_gain_pp=0.0 if method == "edge_only" else quality["primary"]["delta_pp"],
                strict_accuracy_pct=quality["strict"][qprefix+"_accuracy_pct"],
                edge_cost_reduction_pct=(edge_cost-summary["cost_ms_token"])/edge_cost*100, **summary))
        for comparator in ("edge_only", "per_round_ucb_warm", "ucbspec_arm_bonus_warm", "b6_calibrated_fixed"):
            actual = "fixed_depth_%02d" % condition["aliases"][comparator] if comparator in condition["aliases"] else comparator
            identity = ("edge_only", "edge_only") if comparator == "edge_only" else (cid, actual)
            sn, baseline = ratios[(cid, "sn_warm")], ratios[identity]
            comparison = paired_test([sn[s]-baseline[s] for s in sn])
            comparison["mean_paired_cost_reduction_pct"] = float(np.mean([(baseline[s]-sn[s])/baseline[s]*100 for s in sn]))
            contrast_rows.append(dict(condition=cid, comparator=comparator, **comparison))
    for row, adjusted in zip(contrast_rows, holm([r["paired_t_p"] for r in contrast_rows])):
        row["holm_p"] = adjusted
    save_json(result_dir / "audit.json", audit_result)
    save_json(result_dir / "quality.json", quality)
    save_json(result_dir / "quality_pairs.json", pairs)
    save_csv(result_dir / "scored_answers.csv", scored)
    save_csv(result_dir / "table4_all_metrics.csv", table)
    save_json(result_dir / "latency_contrasts.json", contrast_rows)
    save_csv(result_dir / "session_costs.csv", [dict(condition=c, method=m, session=s, cost_ms_token=v)
        for (c, m), values in sorted(ratios.items()) for s, v in values.items()])
    write_report(quality, table, contrast_rows, audit_result)
    write_tex(quality, table)
    print(json.dumps(dict(audit=audit_result, quality=quality, contrasts=contrast_rows), indent=2))


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:.{decimals}f}"


def write_report(quality, table, contrasts, audit_result):
    q, strict = quality["primary"], quality["strict"]
    look = {(r["condition"], r["method"]): r for r in table}
    text = ["# Table 4：公开数据复算", "",
        "本报告分析 32 道题、8 个模型重启会话、576 个已记录请求；离线复算不是重新采集硬件时延。",
        "仅包含结束标记修正后的正式会话，未合并预检或修正前的记录。设计与修正说明见 docs/REPRODUCIBILITY.md 和 docs/TERMINATION_AUDIT.md。",
        "", "## 回答质量及其含义", "",
        f"主评分：纯边缘 {q['edge_correct']}/32（{q['edge_accuracy_pct']:.2f}%）；云端及全部精确协同方法 {q['target_correct']}/32（{q['target_accuracy_pct']:.2f}%）。",
        f"相对纯边缘变化 {q['delta_pp']:+.2f} 个百分点，题目配对 bootstrap 95% CI [{q['delta_ci_pp'][0]:.2f}, {q['delta_ci_pp'][1]:.2f}]；精确 McNemar p={q['exact_mcnemar_p']:.6g}。",
        f"严格无回退评分：纯边缘 {strict['edge_correct']}/32，云端及协同 {strict['target_correct']}/32，变化 {strict['delta_pp']:+.2f} 个百分点；CI [{strict['delta_ci_pp'][0]:.2f}, {strict['delta_ci_pp'][1]:.2f}]。",
        "", "所有 544 个非纯边缘请求均按题核对了完整 token 序列及停止原因，两个网络下全部协同方法与 Target-only 一致。质量变化是目标模型参与带来的交付回答质量变化，所有精确方法共享，不能归因于 SN 独有的选 k 规则，也不代表本地模型权重或内在推理能力被训练增强。", "",
        "边缘使用原 Qwen2.5-0.5B 基础模型，目标为 Qwen2.5-7B-Instruct；质量差异同时包含规模与指令训练差异。此结果只覆盖预先抽取的 32 道短 GSM8K test 题、原裸提示模板和 512-token 上限，不是完整 GSM8K 官方模型得分。", "",
        "## 核心表：所有算法与纯边缘参照", "",
        "成本为各会话总完成时间/总交付 token 的 8 会话等权均值，单位 ms/token，越低越好。质量每题只计一次，不能因网络或算法重复而增大样本量。", "",
        "| 方法 | 正确率 % | 相对纯边缘 pp | Wi-Fi 成本 | 延迟代理成本 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for method, label in LABELS:
        a, b = look[("lan_direct", method)], look[("lan_plus150", method)]
        text.append(f"| {label} | {a['accuracy_pct']:.2f} | {a['accuracy_gain_pp']:+.2f} | {a['cost_ms_token']:.2f} | {b['cost_ms_token']:.2f} |")
    text.append("")
    for cid, path_label in PATH_LABELS.items():
        distributed = [r for r in table if r["condition"] == cid and r["method"] not in ("edge_only", "direct_target")]
        best = min(r["cost_ms_token"] for r in distributed)
        winners = [r["label"] for r in distributed if abs(r["cost_ms_token"]-best) < 1e-9]
        sn, per_round = look[(cid, "sn_warm")], look[(cid, "per_round_ucb_warm")]
        reduction = (per_round["cost_ms_token"]-sn["cost_ms_token"])/per_round["cost_ms_token"]*100
        text.append(f"{path_label}：分布式方法中最低样本均值为 {'、'.join(winners)}（{best:.2f} ms/token）；SN 为 {sn['cost_ms_token']:.2f} ms/token，相对 Per-round UCB 的均值成本降低 {reduction:+.2f}%。这里的最低均值只描述本轮样本，显著性另见预设配对比较，不能写成普遍最优。")
    text += ["", "B1/B2/B3 对应 k=5/1/7；本地 B4/B5/B6 对应 k=1/1/5，代理 B4/B5/B6 对应 k=9/5/9。同一固定 k 的别名复用同一次实测，不是多个独立运行。B3 是原协议的阈值适配，B5 是常数成本几何模型。所有在线控制器收到同一历史 m=27 校准，不用新题调参。", "",
        "本轮复用 09-24 冻结校准，采用 1024-token graph cache 和 512-token 输出上限。这是历史校准迁移到预先选定的题目和长回答的比较，不代表在新协议下重新充分校准后的最佳固定策略；没有重测完整十臂网格，也不能据此声称超过真正的最优固定长度。", "",
        "## 请求时延与通信负担", "",
        "验证 RPC 包含云端计算；下表残差是逐轮 RPC 减服务器验证计时的累计量，仍含协议与端侧开销，不能称纯网络时延。Target-only 没有验证 RPC，因此对应列为 NA；Edge-only 无云调用。载荷仅计应用层请求与响应主体，不含 TCP/IP、Wi-Fi 帧头或请求后清理。", ""]
    for cid, path_label in PATH_LABELS.items():
        text += [f"### {path_label}", "", "| 方法 | E2E s | TTFT ms | 请求 p95 s | 验证 RPC s | 验证残差 s | 生成交互数 | 载荷 KiB |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method, label in LABELS:
            row = look[(cid, method)]
            rpc = row["verification_rpc_ms"]
            residual = row["verification_residual_ms"]
            text.append(f"| {label} | {row['e2e_s']:.2f} | {row['ttft_ms']:.1f} | {row['e2e_p95_s']:.2f} | {fmt(None if rpc is None else rpc/1000)} | {fmt(None if residual is None else residual/1000)} | {row['generation_interactions']:.1f} | {row['application_kib']:.2f} |")
        target = look[(cid, "direct_target")]
        text += ["", f"Target-only 的流式请求减服务器总耗时残差均值为 {target['target_stream_residual_ms']:.2f} ms；流式重叠传输和服务器区间使其与逐轮验证残差不可互换。", ""]
    text += ["请求 p95 是每方法 32 次请求的经验分位数，不是精确的总体尾时延估计。不同模型生成长度和正确率不同，纯边缘与协同的 ms/token 差不能解释为同一输出序列的加速。", "", "## SN 的预设配对比较", "",
        "差值为 SN 减基线，负值有利于 SN。8 次比较整体 Holm 校正，未按结果删减比较。单位为 8 个会话的配对成本，不是把所有请求当独立重复。", "",
        "| 网络 | 基线 | 差值 ms/token | 95% t CI | Holm p | 符号翻转 p（敏感性） |",
        "| --- | --- | ---: | --- | ---: | ---: |"]
    names = dict(LABELS)
    for c in contrasts:
        text.append(f"| {PATH_LABELS[c['condition']]} | {names[c['comparator']]} | {c['mean_difference_ms_token']:+.2f} | [{c['ci_low']:.2f}, {c['ci_high']:.2f}] | {c['holm_p']:.5f} | {c['sign_flip_p']:.5f} |")
    text += ["", "每条件仅 8 个会话，t 区间及检验依赖差值分布的近似假设。符号翻转 p 值未做多重校正，仅作敏感性分析；穷举符号翻转仍要求相应对称/可交换性。两者都不能证明跨设备、跨工作负载的普遍最优。", "", "## 输出与计时审计", "",
        "| 方法 | 平均 token | 答案停止 | EOS 停止 | 达 512 上限 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in ("edge_only", "direct_target", "sn_warm"):
        row = look[("lan_direct", method)]
        text.append(f"| {names[method]} | {row['output_tokens']:.1f} | {row['answer_requests']} | {row['eos_requests']} | {row['cap_requests']} |")
    text += ["", "每个方法使用同一最终数值答案边界规则，检测不看正确答案；如无完整标记，按模型完整的原生 EOS 集合或 512 cap 停止。EOS 不计入交付 token。协同的终轮多验证成本完整计入，但终轮不更新学习器。计时从已 token 化的输入上下文进入生成函数开始，包含 prefill、控制器、生成及通信验证直到完成；输入分词、模型加载、CUDA graph 建立、历史校准取得和完成后的清理不计入请求主时延。原始记录保留这些边界。边缘遥测已归档，本报告不对未完整覆盖的请求推算能耗。", "",
        "本轮采用当前实际本地 Wi-Fi 路径及同路径软件转发延迟（每方向每 TCP 数据块 75 ms），后者不代表真实远端服务，也不等于严格固定 RTT 增加。",
        "", "## 可复核文件", "",
        "- `docs/REPRODUCIBILITY.md`、`docs/STATISTICS.md`：采集与统计设计。",
        "- `table4_all_metrics.csv`：24 行完整指标（Edge-only 跨路径共享 32 次实测）。",
        "- `latency_contrasts.json`、`session_costs.csv`：8 项检验和会话级原值。",
        "- `scored_answers.csv`、`quality_pairs.json`：模型全文回答、提取数值及题目级配对。",
        "- `audit.json`：采集完整性及公开归档校验；原始文件位于 data/table4。",
        "- `data/table4/hardware_run/capability-01..08/`：原始请求、轮次与输出一致性记录。完整遥测和控制器状态不在本公开包中。",
        "- `table4_capability.tex`：独立可编译的新表，原 32/128-token 表不被覆盖。", ""]
    text.extend(["- `table4_insert.tex`：全指标表片段；新版核心列见 build_table4.py 生成的 table4_compact.md / .csv。", ""])
    (OUTPUT / "RESULTS.md").write_text("\n".join(text))


def write_tex(quality, table):
    look = {(r["condition"], r["method"]): r for r in table}
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[margin=1.2cm,landscape]{geometry}",
        r"\usepackage{booktabs,multirow}", r"\begin{document}\setcounter{table}{3}", r"\begin{table}[ht]\centering\small",
        r"\caption{Answer quality and latency on yahboom with at most 512 new tokens per request. All 576 measured requests are included.}",
        r"\begin{tabular}{lrr rrr rrr}", r"\toprule",
        r"& \multicolumn{2}{c}{Task quality ($n=32$)} & \multicolumn{3}{c}{Local Wi-Fi} & \multicolumn{3}{c}{Wi-Fi with delay proxy} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Method & Correct (\%) & Gain (pp) & ms/token & TTFT (ms) & RPC residual (s) & ms/token & TTFT (ms) & RPC residual (s) \\", r"\midrule"]
    for method, label in LABELS:
        a, b = look[("lan_direct", method)], look[("lan_plus150", method)]
        residuals = ["---" if r["verification_residual_ms"] is None else f"{r['verification_residual_ms']/1000:.2f}" for r in (a,b)]
        lines.append(f"{label} & {a['accuracy_pct']:.2f} & {a['accuracy_gain_pp']:+.2f} & {a['cost_ms_token']:.2f} & {a['ttft_ms']:.1f} & {residuals[0]} & {b['cost_ms_token']:.2f} & {b['ttft_ms']:.1f} & {residuals[1]} \\\\")
    q = quality["primary"]
    lines += [r"\bottomrule\end{tabular}", r"\end{table}",
        r"\noindent\footnotesize Costs are means of eight restarted-session time/token ratios (four distinct questions per session). Gain is accuracy minus edge-only accuracy in percentage points; identical target-preserving outputs share the same quality, not independent quality samples. ",
        f"The paired quality change is {q['delta_pp']:+.2f} pp (95\\% paired bootstrap CI [{q['delta_ci_pp'][0]:.2f}, {q['delta_ci_pp'][1]:.2f}]). ",
        r"RPC residual sums verification RPC minus reported server verification time; it includes protocol overhead and is not pure network latency. Dashes mean no verification RPC. Edge-only has no cloud traffic. ",
        r"Edge-only is measured once per question and shared across the two path comparisons. ",
        r"The small model is Qwen2.5-0.5B Base and the target is Qwen2.5-7B-Instruct. Quality gains reflect collaboration with this target and include model-size/instruction-training differences, not improved local weights or a unique SN accuracy benefit. ",
        r"Generation stops at a completed explicit numeric answer, any native model EOS token, or the 512-token ceiling. EOS tokens are excluded from delivered-token counts. ",
        r"All online methods share the frozen $m=27$ calibration. B1/B2/B3 use $k=5/1/7$; B4/B5/B6 use $1/1/5$ locally and $9/5/9$ with the proxy. Identical fixed-depth aliases reuse one measured run. Calibration acquisition and model/graph initialization are excluded. ",
        r"This follow-up uses answer completion rather than the original fixed 32/128-token protocol; its values must not be silently substituted into that table.", r"\end{document}"]
    (OUTPUT / "table4_capability.tex").write_text("\n".join(lines)+"\n")
    tab_start = next(i for i, line in enumerate(lines) if line.startswith(r"\begin{tabular}"))
    tab_end = lines.index(r"\bottomrule\end{tabular}")
    table_end = lines.index(r"\end{table}")
    notes = "\n".join(lines[table_end+1:-1]).replace(r"\noindent\footnotesize", "")
    fragment = [r"\begin{table*}[t]\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                next(line for line in lines if line.startswith(r"\caption")),
                r"\label{tab:capability_followup}"]
    fragment += lines[tab_start:tab_end+1]
    fragment += [r"\par\smallskip\parbox{\textwidth}{\scriptsize", notes, "}", r"\end{table*}"]
    (OUTPUT / "table4_insert.tex").write_text("\n".join(fragment)+"\n")


if __name__ == "__main__":
    main()
