#!/usr/bin/env bash
# E0 数据与标签管线(manifest / 划分 / 审计)
# 对应计划 §E0。全部 CPU 可跑,无 GPU 依赖。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== E0.1 构建 manifest v2(补 PMA/GA/BW/sex/exam_date/eye/device/series)=="
python3 -m neoropfm.data.build_manifest_v2

echo "== E0.2 LODO 划分质检(split 单位/患者重叠/类别平衡)=="
python3 -m neoropfm.data.check_splits

echo "== E0.3 SSL 无标签语料 manifest(E2 用;含辅助数据集,hvdropdb/macretina/vessel/OD/plos)=="
# [待实现] python3 -m neoropfm.data.build_ssl_corpus_manifest
echo "  [待实现] build_ssl_corpus_manifest(4 主数据集 10,656 + 辅助 2,821,排除 coph100)"

echo "== E0.4 标签规则敏感性材料(附件稿件标签规则核对,RIDIRP DG/PF 编码核对)=="
# [待实现] 附件稿件定位后:python3 -m neoropfm.data.label_rule_sensitivity
echo "  [待实现] label_rule_sensitivity(等待用户提供附件稿件原文)"
