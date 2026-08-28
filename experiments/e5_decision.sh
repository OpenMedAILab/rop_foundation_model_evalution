#!/usr/bin/env bash
# E5 临床决策价值分析与结果汇总
# 对应计划 §E5:DCA、工作负荷模拟、results_summary.md(回答方案 Q1-Q5)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== E5.1 决策曲线分析(全部模型,DCA 对比 treat-all / treat-none)=="
# [待实现] python3 -m neoropfm.eval.dca --config configs/dca.yaml
echo "  [待实现] dca(净收益曲线,按测试折汇总)"

echo "== E5.2 筛查工作负荷模拟(锁定阈值下漏诊/转诊数)=="
# [待实现] python3 -m neoropfm.eval.workload_simulation --config configs/workload.yaml
echo "  [待实现] workload_simulation(每 1000 婴儿转诊量/漏诊数)"

echo "== E5.3 结果汇总(回答方案 Q1-Q5)=="
# [待实现] python3 -m neoropfm.report.results_summary --config configs/report.yaml
echo "  [待实现] results_summary → outputs/results_summary.md"
