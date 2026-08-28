"""E5 重构 —— ≥99% 灵敏度锁定操作点 + 校准/风险覆盖指标。

原 E5 用 sens95 阈值却称"保证不漏诊"不当;且缺 NPV/校准/ECE/AURC。
本脚本改为:
1. 操作点锁定:每折用 train 侧 OOF 预测锁阈值 θ99(允许错过的阳性 ≤1%,即 train
   灵敏度 ≥99%)与 θ100(train 灵敏度 100%)。
   **主口径 = 访视级**(决策单元为单次筛查检查,与锁定表 locked_metrics 同口径);
   同时并排产出**患者级口径**(同患者全部访次 mean-p、max-y 池化)作为口径敏感性警示——
   该口径在本数据上 AUROC 虚高(如 ridirp 测试折仅 5 名阳性患者 × 人均约 175 次访视,
   池化后 AUROC≈1.0)且 θ99 特异性塌缩,不作部署口径。
2. 一次性 test 评估:各折 test 按本折 θ99/θ100 决策,报告:灵敏度/特异度/PPV/NPV、
   灵敏度 bootstrap CI(2,000, seed 42,患者级重采样)、最差中心(单数据集折)灵敏度、
   Brier、ECE(10 分位 bin)、AURC(margin 风险-覆盖曲线,coverage 1.0→0.6 每 0.02 积分);
3. 安全自动化比例:margin τ ∈ {0, .05, …, .5} 不确定访视交人工(人工假定正确),
   管线灵敏度 = 1 − (自动放走的阳性)/n_pos;选最小 τ 使管线灵敏度 ≥99%,
   报告该 τ 下的自动化比例与自动子集灵敏度/特异度。

口径:仅使用 outputs/checkpoint_iso/{model}_iso/fold_{held}_predictions.csv
  (R3 隔离协议产物:train=OOF、test=一次性),单权重基线同口径。
模型:outputs/checkpoint_iso 下全部 *_iso 目录。
输出:outputs/e5_locked/{model}_e5_locked.csv(逐折 × 口径)+ all_models_e5_locked.csv(汇总)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ISO = REPO / "outputs/checkpoint_iso"
OUT = REPO / "outputs/e5_locked"
from neoropfm.common import HELDOUTS  # noqa: E402


def pool(df: pd.DataFrame, patient: bool) -> pd.DataFrame:
    """patient=True:患者级聚合(同患者多图 mean-p,标签 max);否则保持访视级。"""
    if not patient:
        return df.copy()
    return df.groupby(["dataset", "patient_id"]).agg(y=("y", "max"), p=("p", "mean")).reset_index()


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """等分位 bin ECE(10 bin,加权 |conf−acc|)。"""
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9
    idx = np.digitize(p, bins) - 1
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(total)


def sens_ci(y, p, unit, theta, n_boot=2000, seed=42):
    def _sens(yy, pp):
        return float((pp >= theta).astype(int)[yy == 1].mean()) if (yy == 1).sum() else float("nan")
    return patient_level_bootstrap(y, p, unit, _sens, n_boot, seed)


def aurc(y: np.ndarray, p: np.ndarray, cov_start: float = 1.0, cov_end: float = 0.6,
         step: float = 0.02) -> float:
    """margin 风险-覆盖曲线下面积(cover ∈ [0.6, 1.0])。"""
    u = 1 - np.maximum(p, 1 - p)
    order = np.argsort(u)
    vals = []
    for cov in np.arange(cov_start, cov_end - 1e-9, -step):
        k = max(1, int(round(cov * len(y))))
        keep = order[:k]
        if len(np.unique(y[keep])) < 2 or y[keep].sum() < 3:
            continue
        vals.append(roc_auc_score(y[keep], p[keep]))
    return float(np.mean(vals)) if vals else float("nan")


def run_model(iso_dir: Path) -> None:
    model = iso_dir.name[:-4]
    rows = []
    for held in HELDOUTS:
        f = iso_dir / f"fold_{held}_predictions.csv"
        if not f.exists():
            continue
        pred = pd.read_csv(f)
        for level in ("visit", "patient"):
            patient = level == "patient"
            tr = pool(pred[pred["subset"] == "train"], patient)
            te = pool(pred[pred["subset"] == "test"], patient)
            pos = np.sort(tr.loc[tr["y"] == 1, "p"].to_numpy())
            if len(pos) == 0 or len(te) < 5:
                continue
            n_miss = int(np.floor(0.01 * len(pos)))          # 允许错过的阳性数
            th99 = pos[n_miss]                                # train 灵敏度 ≥ 99%
            th100 = pos[0]                                    # train 灵敏度 = 100%

            unit = (te["dataset"] + "|" + te["patient_id"]).to_numpy()
            y, p = te["y"].to_numpy(), te["p"].to_numpy()
            n_pos = int(y.sum())
            for name, th in (("theta99", th99), ("theta100", th100)):
                dec = (p >= th).astype(int)
                sens = float(dec[y == 1].mean()) if n_pos else float("nan")
                spec = float((1 - dec)[y == 0].mean()) if (y == 0).sum() else float("nan")
                ppv = float(dec[y == 1].sum() / dec.sum()) if dec.sum() else float("nan")
                npv = float((1 - dec)[y == 0].sum() / (1 - dec).sum()) if (1 - dec).sum() else float("nan")
                ci = sens_ci(y, p, unit, th)
                rows.append({
                    "model": model, "level": level, "heldout_dataset": held, "op": name,
                    "threshold": th, "train_sens": 1 - n_miss / len(pos) if name == "theta99" else 1.0,
                    "test_sens": sens, "test_sens_lo": ci["lower"], "test_sens_hi": ci["upper"],
                    "test_spec": spec, "ppv": ppv, "npv": npv,
                    "n_test_units": len(te), "n_test_pos": n_pos,
                })

            # 安全自动化:最小 τ 使管线灵敏度 ≥99%
            u = 1 - np.maximum(p, 1 - p)
            chosen = None
            for tau in np.arange(0.0, 0.51, 0.05):
                certain = u <= tau
                missed = ((p < th99) & (y == 1) & certain).sum()
                pipe_sens = 1 - missed / n_pos if n_pos else float("nan")
                auto_share = float(certain.mean())
                if pipe_sens >= 0.99:
                    chosen = {"margin_tau": tau, "pipeline_sens": pipe_sens,
                              "auto_share": auto_share,
                              "auto_sens": float((p[certain & (y == 1)] >= th99).mean())
                              if (certain & (y == 1)).sum() else float("nan"),
                              "auto_spec": float((p[certain & (y == 0)] < th99).mean())
                              if (certain & (y == 0)).sum() else float("nan"),
                              "missed_pos": int(missed)}
                    break
            if chosen is None:
                chosen = {"margin_tau": float("nan"), "pipeline_sens": float("nan"),
                          "auto_share": float("nan"), "auto_sens": float("nan"),
                          "auto_spec": float("nan"), "missed_pos": 0}
            rows.append({"model": model, "level": level, "heldout_dataset": held, "op": "safe_auto",
                         "threshold": th99, "train_sens": 1 - n_miss / len(pos),
                         "test_sens": chosen["pipeline_sens"],
                         "test_sens_lo": float("nan"), "test_sens_hi": float("nan"),
                         "test_spec": float("nan"), "ppv": float("nan"), "npv": float("nan"),
                         "n_test_units": len(te), "n_test_pos": n_pos,
                         **{k: v for k, v in chosen.items() if k != "pipeline_sens"},
                         "pipeline_sens": chosen["pipeline_sens"]})
            # 全量校准与风险覆盖(与操作点无关)
            rows.append({"model": model, "level": level, "heldout_dataset": held, "op": "calib_aurc",
                         "threshold": float("nan"), "train_sens": float("nan"),
                         "test_sens": float("nan"), "test_sens_lo": float("nan"),
                         "test_sens_hi": float("nan"), "test_spec": float("nan"),
                         "ppv": float("nan"), "npv": float("nan"),
                         "n_test_units": len(te), "n_test_pos": n_pos,
                         "brier": brier_score_loss(y, p), "ece": ece(y, p),
                         "aurc": aurc(y, p), "auroc_full": roc_auc_score(y, p)})
            print(f"[{model}] {held}/{level}: θ99={th99:.4f} (train_sens={1-n_miss/len(pos):.3f}) "
                  f"test_sens={rows[-3]['test_sens']:.3f} spec={rows[-3]['test_spec']:.3f} "
                  f"safe_auto={rows[-2]['auto_share']:.3f} brier={rows[-1]['brier']:.3f} "
                  f"ece={rows[-1]['ece']:.3f} aurc={rows[-1]['aurc']:.3f}",
                  flush=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{model}_e5_locked.csv", index=False)
        print(f"[{model}] saved → {OUT / f'{model}_e5_locked.csv'}", flush=True)


def aggregate() -> None:
    rows = []
    for f in sorted(OUT.glob("*_e5_locked.csv")):
        df = pd.read_csv(f)
        if "level" not in df.columns:
            continue  # 跳过旧格式(含汇总文件自身)<｜end▁of▁thinking｜>
        for (op, level), sub in df.groupby(["op", "level"]):
            row = {"model": sub["model"].iloc[0], "op": op, "level": level,
                   "n_folds": len(sub), "n_test_units": int(sub["n_test_units"].sum()),
                   "n_test_pos": int(sub["n_test_pos"].sum())}
            if op == "theta99":
                row["mean_test_sens"] = sub["test_sens"].mean()
                row["min_test_sens"] = sub["test_sens"].min()
                row["worst_fold"] = sub.loc[sub["test_sens"].idxmin(), "heldout_dataset"]
                row["mean_test_spec"] = sub["test_spec"].mean()
                row["mean_npv"] = sub["npv"].mean()
                row["mean_ppv"] = sub["ppv"].mean()
            elif op == "theta100":
                row["mean_test_sens"] = sub["test_sens"].mean()
                row["min_test_sens"] = sub["test_sens"].min()
                row["worst_fold"] = sub.loc[sub["test_sens"].idxmin(), "heldout_dataset"]
            elif op == "safe_auto":
                row["mean_pipeline_sens"] = sub["pipeline_sens"].mean()
                row["mean_auto_share"] = sub["auto_share"].mean()
                row["mean_margin_tau"] = sub["margin_tau"].mean()
            else:  # calib_aurc
                row["brier"] = sub["brier"].mean()
                row["ece"] = sub["ece"].mean()
                row["aurc"] = sub["aurc"].mean()
                row["auroc_full"] = sub["auroc_full"].mean()
            rows.append(row)
    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT / "all_models_e5_locked.csv", index=False)
    print("汇总 →", OUT / "all_models_e5_locked.csv")


def main() -> None:
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in ISO.iterdir() if d.is_dir() and d.name.endswith("_iso"))
    print(f"模型目录: {[d.name for d in dirs]}", flush=True)
    if "aggregate" not in only:
        for d in dirs:
            run_model(d)
    aggregate()


if __name__ == "__main__":
    main()
