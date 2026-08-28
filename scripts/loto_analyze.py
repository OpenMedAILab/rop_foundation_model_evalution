"""LOTO(留一数据集继续预训练)敏感性分析:数据落地后运行。

对每个被剔除数据集 f,f 的 LOTO 模型 = outputs/ssl/{run_prefix}_{f}[/{suffix}]
的**最优 checkpoint**(按该 run 的 checkpoint_selection.csv,4 折 LODO 探针均值,
与 E2 主协议同口径),与全语料模型逐折对比:

- **直接泄漏检验**(heldout == 被剔除数据集):若 RIDIRP 折 +0.189 的大部来自
  "见过无标签测试图",该行 Δ 应显著回落;
- **语料构成效应**(其余 heldout):剔除后其余折语料占比上升,观察 Δ。

输出:outputs/aggregate_e2/{out}(默认 loto_comparison.csv)
列:excluded_dataset, heldout_dataset, full_auc [CI], loto_auc [CI], delta, p_delong

泛化(--run-prefix/--full-run-dir/--out):v1 路线与 heads 路线各自成表,
对照 = 对应全语料 run 的 checkpoint_selection.csv 最优(与 LOTO run 同口径)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402
from neoropfm.stats.delong import delong_test  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
from neoropfm.common import HELDOUTS  # noqa: E402
PROBE_DIR = REPO / "outputs/probes"


def best_key(run_dir: Path) -> str:
    """按该 run 的 checkpoint_selection.csv(与 E2 主协议同口径:4 折 LODO 探针均值)
    选最优 checkpoint。"""
    sel = pd.read_csv(run_dir / "checkpoint_selection.csv")
    return str(sel.sort_values("mean_auroc", ascending=False).iloc[0]["model_key"])


def loto_run_dir(prefix: str, excluded: str) -> Path | None:
    """定位 LOTO run 目录:前缀_{excluded}(可带后缀,如 minus_szeh_irops_rerun_20260823)。
    多个候选(且都有选择结果)时报错;否则取唯一有 checkpoint_selection.csv 者。"""
    cands = sorted(
        p for p in (REPO / "outputs/ssl").glob(f"{prefix}_{excluded}*")
        if (p / "checkpoint_selection.csv").exists()
    )
    if not cands:
        return None
    if len(cands) > 1:
        raise RuntimeError(f"{prefix}_{excluded}* 多个候选 run: {[c.name for c in cands]}")
    return cands[0]


def loto_key(prefix: str, excluded: str) -> str:
    """LOTO run 的最优 checkpoint key(各 run 独立选择;训练链会为最终检查点
    全部产出探针,故所选 key 的探针必然存在)。"""
    return best_key(loto_run_dir(prefix, excluded))


def read_metrics(key: str, held: str) -> dict:
    csv = PROBE_DIR / key / f"{key}_probe_lodo_metrics.csv"
    m = pd.read_csv(csv).set_index("heldout_dataset")
    auroc = float(m.loc[held, "auroc"])
    if "auroc_lo" in m.columns:
        return {"auroc": auroc,
                "lo": float(m.loc[held, "auroc_lo"]),
                "hi": float(m.loc[held, "auroc_hi"])}
    # 训练链未附带 CI 时,按协议现算(患者级 bootstrap 2,000 次,seed 42)
    pred = pd.read_csv(PROBE_DIR / key / f"fold_{held}_predictions.csv")
    pred = pred[pred["subset"] == "test"]
    ci = patient_level_bootstrap(
        pred["y"].to_numpy(), pred["p"].to_numpy(),
        pred["patient_id"].to_numpy(), roc_auc_score, n_boot=2000, seed=42)
    return {"auroc": auroc, "lo": ci["lower"], "hi": ci["upper"]}


def paired_delong(key_a: str, key_b: str, held: str) -> dict:
    a = pd.read_csv(PROBE_DIR / key_a / f"fold_{held}_predictions.csv")
    b = pd.read_csv(PROBE_DIR / key_b / f"fold_{held}_predictions.csv")
    a = a[a["subset"] == "test"].set_index("sample_id")
    b = b[b["subset"] == "test"].set_index("sample_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y = a["y"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"diff": np.nan, "p": np.nan}
    return delong_test(y, a["p"].to_numpy(), b["p"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="loto_ibot_dinov2s_minus",
                    help="LOTO run 目录前缀(默认 v1 路线;heads 用 loto_ibot_dinov2s_heads_minus)")
    ap.add_argument("--full-run-dir", default="outputs/ssl/ibot_dinov2s_v1",
                    help="全语料对照 run 目录(其 checkpoint_selection.csv 最优为 FULL_KEY)")
    ap.add_argument("--out", default="loto_comparison.csv",
                    help="输出文件名(写 outputs/aggregate_e2/ 下)")
    args = ap.parse_args()
    full_run = REPO / args.full_run_dir
    if not (full_run / "checkpoint_selection.csv").exists():
        print(f"[skip] 全语料 run {full_run.name} 尚无 checkpoint 选择结果", flush=True)
        return
    full_key = best_key(full_run)
    rows = []
    for excluded in HELDOUTS:
        run_dir = loto_run_dir(args.run_prefix, excluded)
        if run_dir is None:
            print(f"[skip] {args.run_prefix}_{excluded}* 训练中(尚无 checkpoint 选择结果)", flush=True)
            continue
        key = loto_key(args.run_prefix, excluded)
        if not (PROBE_DIR / key).exists():
            print(f"[skip] {key} 探针尚未产出", flush=True)
            continue
        for held in HELDOUTS:
            full = read_metrics(full_key, held)
            loto = read_metrics(key, held)
            d = paired_delong(full_key, key, held)
            rows.append({
                "excluded_dataset": excluded, "heldout_dataset": held,
                "full_auroc": full["auroc"], "full_lo": full["lo"], "full_hi": full["hi"],
                "loto_auroc": loto["auroc"], "loto_lo": loto["lo"], "loto_hi": loto["hi"],
                "delta": loto["auroc"] - full["auroc"],
                "p_delong": d["p"], "delong_diff": d["diff"],
            })
    if not rows:
        print("LOTO 数据未就绪,退出", flush=True)
        return
    df = pd.DataFrame(rows)
    out = REPO / "outputs/aggregate_e2" / args.out
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("=== 直接泄漏检验(heldout == 被剔除数据集)===", flush=True)
    key_rows = df[df.excluded_dataset == df.heldout_dataset]
    print(key_rows[["excluded_dataset", "full_auroc", "loto_auroc", "delta", "p_delong"]]
          .round(4).to_string(index=False), flush=True)
    print("\n=== 全部 16 行(heldout × excluded)===", flush=True)
    print(df[["excluded_dataset", "heldout_dataset", "full_auroc",
              "loto_auroc", "delta", "p_delong"]].round(4).to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
