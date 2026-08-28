"""R9 配套:基于 R3 隔离预测的配对 bootstrap 差异检验(替代旧硬编码 DeLong 表)。

口径:outputs/checkpoint_iso/{model}_iso/fold_{held}_predictions.csv 的 test 子集,
访视级(点估与锁定表 locked_metrics 同口径——患者级全访次池化在 ridirp 等折上
因"5 名阳性患者 × 人均 175 次访视"而方差坍缩至 AUROC≈1,不作口径),
重采样单元为患者(同患者同进同出),2,000 次 bootstrap(seed 42)。
对比集(与旧 表 6 一致,但用 R3 每折隔离选点):
  v1_iso vs dinov2_vits14 / v1_iso vs retfound_green /
  heads_iso vs retfound_green / heads_iso vs v1_iso / mae_iso vs retfound_green
输出:outputs/checkpoint_iso/paired_tests.csv(逐对比 × 逐折 diff/CI/p)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import HELDOUTS, parse_heldouts  # noqa: E402
from neoropfm.stats.bootstrap import fold_mean_delta_bootstrap, paired_bootstrap_diff  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ISO = REPO / "outputs/checkpoint_iso"
PAIRS = [
    ("ibot_dinov2s_v1_iso", "dinov2_vits14_iso", "NeoROP-FM(iBOT) vs 起点 DINOv2-S"),
    ("ibot_dinov2s_v1_iso", "retfound_green_iso", "NeoROP-FM(iBOT) vs RETFound-Green"),
    ("ibot_dinov2s_heads_iso", "retfound_green_iso", "NeoROP-FM(+辅助头) vs RETFound-Green"),
    ("ibot_dinov2s_heads_iso", "ibot_dinov2s_v1_iso", "NeoROP-FM(+辅助头) vs NeoROP-FM(iBOT)"),
    ("mae_retfound_green_v1_iso", "retfound_green_iso", "MAE 备选 vs RETFound-Green"),
    # P2 消融/打乱对照(模型由 e2c 训练,缺文件时自动跳过;英文名——直接进 S16/正文消融句)
    ("ibot_dinov2s_heads_pma_only_iso", "ibot_dinov2s_v1_iso", "PMA-only ablation vs. iBOT"),
    ("ibot_dinov2s_heads_cons_only_iso", "ibot_dinov2s_v1_iso", "visit-only ablation vs. iBOT"),
    ("ibot_dinov2s_heads_iso", "ibot_dinov2s_heads_pma_only_iso", "heads vs. PMA-only"),
    ("ibot_dinov2s_heads_iso", "ibot_dinov2s_heads_cons_only_iso", "heads vs. visit-only"),
    ("ibot_dinov2s_heads_iso", "ibot_dinov2s_heads_shuf_pma_iso", "heads vs. shuffled-PMA control"),
    ("ibot_dinov2s_heads_iso", "ibot_dinov2s_heads_shuf_visit_iso", "heads vs. shuffled-visit control"),
]
# 消融族(独立 Holm 校正族,6 个预指定对比;不并入主比较的 5 对)
ABL_PAIRS = PAIRS[5:]


def load_rows(model: str, held: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """访视级 test 行 y/p + 患者重采样单元(不池化——点估与锁定表一致)。"""
    pred = pd.read_csv(ISO / model / f"fold_{held}_predictions.csv")
    te = pred[pred["subset"] == "test"]
    unit = (te["dataset"] + "|" + te["patient_id"]).to_numpy()
    return te["y"].to_numpy(), te["p"].to_numpy(), unit


def main() -> None:
    rows = []
    for a, b, name in PAIRS:
        for held in HELDOUTS:
            fa, fb = ISO / a / f"fold_{held}_predictions.csv", ISO / b / f"fold_{held}_predictions.csv"
            if not (fa.exists() and fb.exists()):
                print(f"[skip] {name} @ {held}(预测文件缺失)")
                continue
            ya, pa, ua = load_rows(a, held)
            yb, pb, _ = load_rows(b, held)
            r = paired_bootstrap_diff(ya, pa, pb, ua, roc_auc_score, 2000, 42)
            rows.append({"comparison": name, "heldout_dataset": held,
                         "auroc_a": roc_auc_score(ya, pa), "auroc_b": roc_auc_score(yb, pb),
                         "diff": r["diff"], "diff_lo": r["lower"], "diff_hi": r["upper"],
                         "p": r["p"]})
            print(f"{name} @ {held}: {roc_auc_score(ya, pa):.4f} vs {roc_auc_score(yb, pb):.4f} "
                  f"Δ={r['diff']:+.4f} [{r['lower']:+.3f},{r['upper']:+.3f}] p={r['p']:.4f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ISO / "paired_tests.csv", index=False)
    print("saved →", ISO / "paired_tests.csv")


def run_overall() -> None:
    """总体差值(跨 4 折均值)+ Holm 校正(预指定 PAIRS 清单,多重比较控制)。

    输出:outputs/checkpoint_iso/overall_paired_tests.csv
    """
    rows = []
    for a, b, name in PAIRS:
        folds = []
        for held in HELDOUTS:
            fa, fb = ISO / a / f"fold_{held}_predictions.csv", ISO / b / f"fold_{held}_predictions.csv"
            if not (fa.exists() and fb.exists()):
                break
            ya, pa, ua = load_rows(a, held)
            yb, pb, _ = load_rows(b, held)
            folds.append({"y": ya, "p1": pa, "p2": pb, "units": ua})
        if len(folds) < len(HELDOUTS):
            print(f"[skip overall] {name}(缺 {len(HELDOUTS) - len(folds)} 折)")
            continue
        r = fold_mean_delta_bootstrap(folds, n_boot=2000, seed=42)
        rows.append({"comparison": name, "n_folds": r["n_folds"],
                     "diff": round(r["diff"], 4), "diff_lo": round(r["lower"], 4),
                     "diff_hi": round(r["upper"], 4), "p": round(r["p"], 4),
                     "p_holm": None})
        print(f"[overall] {name}: Δ={r['diff']:+.4f} [{r['lower']:+.4f},{r['upper']:+.4f}] "
              f"p={r['p']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        ps = df["p"].to_numpy(float)
        order = np.argsort(ps)
        m = len(ps)
        adj_sorted = np.minimum(1.0, np.sort(ps) * np.arange(m, 0, -1))  # Holm: p_(i)·(m−i+1)
        adj_sorted = np.maximum.accumulate(adj_sorted)  # 单调性
        adj = np.empty(m)
        adj[order] = adj_sorted  # 映射回原行序
        df["p_holm"] = adj.round(4)
    df.to_csv(ISO / "overall_paired_tests.csv", index=False)
    print("saved →", ISO / "overall_paired_tests.csv")
    print(df.to_string(index=False))


def run_ablation() -> None:
    """消融族专属输出(P2;与主比较 5 对分开,各自独立 Holm 族,锁定数字不互相污染)。

    输出:
      outputs/checkpoint_iso/ablation_paired_tests.csv  逐折(comparison, heldout_dataset,
                                                       diff, diff_lo, diff_hi, p)→ S16
      outputs/checkpoint_iso/ablation_overall.csv      总体(pair, delta, ci_lo, ci_hi,
                                                       p, p_holm)→ 正文消融句
    """
    rows = []
    for a, b, name in ABL_PAIRS:
        for held in HELDOUTS:
            fa = ISO / a / f"fold_{held}_predictions.csv"
            fb = ISO / b / f"fold_{held}_predictions.csv"
            if not (fa.exists() and fb.exists()):
                print(f"[skip] {name} @ {held}(预测文件缺失)")
                continue
            ya, pa, ua = load_rows(a, held)
            yb, pb, _ = load_rows(b, held)
            r = paired_bootstrap_diff(ya, pa, pb, ua, roc_auc_score, 2000, 42)
            rows.append({"comparison": name, "heldout_dataset": held,
                         "auroc_a": roc_auc_score(ya, pa), "auroc_b": roc_auc_score(yb, pb),
                         "diff": r["diff"], "diff_lo": r["lower"], "diff_hi": r["upper"],
                         "p": r["p"]})
    df = pd.DataFrame(rows)
    if len(df) < len(ABL_PAIRS) * len(HELDOUTS):
        print(f"[ablation] 仅 {len(df)}/{len(ABL_PAIRS) * len(HELDOUTS)} 行——"
              "队列尚未全部完成,不写输出(防部分门控误开)")
        return
    df.to_csv(ISO / "ablation_paired_tests.csv", index=False)
    print("saved →", ISO / "ablation_paired_tests.csv")

    ov = []
    for a, b, name in ABL_PAIRS:
        folds = []
        for held in HELDOUTS:
            fa = ISO / a / f"fold_{held}_predictions.csv"
            fb = ISO / b / f"fold_{held}_predictions.csv"
            if not (fa.exists() and fb.exists()):
                break
            ya, pa, ua = load_rows(a, held)
            yb, pb, _ = load_rows(b, held)
            folds.append({"y": ya, "p1": pa, "p2": pb, "units": ua})
        if len(folds) < len(HELDOUTS):
            print(f"[skip overall-abl] {name}(缺 {len(HELDOUTS) - len(folds)} 折)")
            continue
        r = fold_mean_delta_bootstrap(folds, n_boot=2000, seed=42)
        ov.append({"pair": name, "delta": round(r["diff"], 4),
                   "ci_lo": round(r["lower"], 4), "ci_hi": round(r["upper"], 4),
                   "p": round(r["p"], 4), "p_holm": None})
    odf = pd.DataFrame(ov)
    if len(odf) < len(ABL_PAIRS):
        print(f"[ablation-overall] 仅 {len(odf)}/{len(ABL_PAIRS)} 对比——不写输出")
        return
    if len(odf):
        ps = odf["p"].to_numpy(float)
        order = np.argsort(ps)
        m = len(ps)
        adj_sorted = np.minimum(1.0, np.sort(ps) * np.arange(m, 0, -1))
        adj_sorted = np.maximum.accumulate(adj_sorted)
        adj = np.empty(m)
        adj[order] = adj_sorted
        odf["p_holm"] = adj.round(4)
        odf.to_csv(ISO / "ablation_overall.csv", index=False)
        print("saved →", ISO / "ablation_overall.csv")
        print(odf.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overall", action="store_true",
                    help="总体差值(跨折均值)+ Holm;缺省=逐折配对检验")
    ap.add_argument("--ablation", action="store_true",
                    help="消融族(6 对比)逐折 + 总体(独立 Holm 族)")
    args = ap.parse_args()
    if args.overall:
        run_overall()
    elif args.ablation:
        run_ablation()
    else:
        main()
