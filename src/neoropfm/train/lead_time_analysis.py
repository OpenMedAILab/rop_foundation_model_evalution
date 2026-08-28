"""Lead-time 分析(E3.5):纵向模型能提前多久识别高风险婴儿。

对于每个事件样本(index 阴性→未来阳性),分析:
- 模型在 index 访视时是否已给出高概率
- 提前量 = 阳性访视 PMA/日期 - index 访视 PMA/日期
- 按提前量分层评估灵敏度

运行:
  python -m neoropfm.train.lead_time_analysis
  python -m neoropfm.train.lead_time_analysis --feature-model retfound_green
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import OUTPUTS_DIR  # noqa: E402

LONG_DIR = OUTPUTS_DIR / "longitudinal"


def analyze_predictions(name: str, preds: pd.DataFrame, manifest: pd.DataFrame, p_col: str = "p"):
    """分析单个模型的 lead-time。"""
    # 只取测试集预测(preds 中每个样本在一个 fold 中)
    # 去重: 每个 sample 只保留一条预测
    preds = preds.drop_duplicates(subset=["patient_id"], keep="first")

    # 与 manifest 关联
    events = preds[preds.y == 1].copy()
    events = events.merge(
        manifest[["patient_id", "dataset", "index_time", "next_time"]],
        on="patient_id", how="left",
    )

    # 计算提前量(周)
    lead_times = []
    for _, row in events.iterrows():
        if row.dataset == "ridirp":
            try:
                lead = float(row.next_time) - float(row.index_time)
            except (ValueError, TypeError):
                lead = np.nan
        else:
            try:
                t_idx = pd.to_datetime(row.index_time)
                t_next = pd.to_datetime(row.next_time)
                lead = (t_next - t_idx).days / 7.0
            except Exception:
                lead = np.nan
        lead_times.append(lead)
    events["lead_weeks"] = lead_times
    events = events.dropna(subset=["lead_weeks"])

    print(f"\n=== {name} ===")
    print(f"Event samples: {len(events)}")
    if len(events) == 0:
        return events
    print(f"Lead time (weeks): mean={events.lead_weeks.mean():.1f}, "
          f"median={events.lead_weeks.median():.1f}, "
          f"range=[{events.lead_weeks.min():.1f}, {events.lead_weeks.max():.1f}]")

    # 用 95% 灵敏度阈值(训练折锁定的近似: 预测概率的 10 分位数)
    th = np.quantile(preds[p_col].values, 0.10)
    print(f"\nDetection by lead time (threshold p>{th:.3f}):")
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    for lo, hi in bins:
        mask = (events.lead_weeks > lo) & (events.lead_weeks <= hi)
        n = mask.sum()
        if n > 0:
            detected = (events.loc[mask, p_col] > th).sum()
            print(f"  {lo}-{hi} weeks: {detected}/{n} detected ({detected/n*100:.0f}%)")

    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-model", default="retfound_green")
    args = ap.parse_args()

    manifest = pd.read_csv(LONG_DIR / f"longitudinal_manifest_{args.feature_model}.csv")

    all_events = []

    # 1. 基线模型
    baseline_path = LONG_DIR / f"baseline_predictions_{args.feature_model}.csv"
    if baseline_path.exists():
        preds = pd.read_csv(baseline_path)
        for model_name in ["longitudinal_multimodal", "static_multimodal", "clinical_only"]:
            sub = preds[preds.model == model_name]
            if len(sub) > 0:
                events = analyze_predictions(model_name, sub, manifest, p_col="p")
                if len(events) > 0:
                    events["model"] = model_name
                    all_events.append(events)

    # 2. Temporal Transformer
    tt_path = LONG_DIR / f"temporal_transformer_predictions_{args.feature_model}.csv"
    if tt_path.exists():
        preds = pd.read_csv(tt_path)
        events = analyze_predictions("temporal_transformer", preds, manifest, p_col="p_mean")
        if len(events) > 0:
            events["model"] = "temporal_transformer"
            all_events.append(events)

    if all_events:
        out = pd.concat(all_events, ignore_index=True)
        out_path = LONG_DIR / f"lead_time_analysis_{args.feature_model}.csv"
        out.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
