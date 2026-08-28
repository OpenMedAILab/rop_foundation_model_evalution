"""构建纵向预测样本(E3.1)。

从 manifest v2 + E1 冻结特征缓存构建访次级纵向数据集:
- RIDIRP: 按 (patient_id, series) 聚合成访次, PMA(周) 为时间轴
- ROP-VL: 按 (patient_id, exam_date) 聚合成访次, 检查日期为时间轴
- 每个访次的多张图像 → 特征均值(visit embedding)
- 标签:
    y_next     : 下一次访视是否 strict_binary 阳性(主终点)
    y_eventual : 任何后续访视是否阳性(辅助终点)
- 排除: 治疗后访视(DG9 / laser-treated)不作为 index 访次,
  但可作为历史上下文(标记为 post_treatment)
- 临床变量: GA / BW / sex / PMA(或检查间隔)

输出:
    outputs/longitudinal/longitudinal_manifest.csv  每个 index 访次一行
    outputs/longitudinal/visit_sequences.npz        每个 index 的特征序列 + 掩码

运行:
    python -m neoropfm.data.build_longitudinal_samples
    python -m neoropfm.data.build_longitudinal_samples --feature-model retfound_green
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V2, OUTPUTS_DIR  # noqa: E402

LONG_DIR = OUTPUTS_DIR / "longitudinal"
FEAT_DIR = OUTPUTS_DIR / "features"

# 治疗后 DG 码(RIDIRP DG9 = status post ROP; ROP-VL Laser-treated 已在 strict 中排除)
POST_TREATMENT_DG = {9.0}


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_V2)
    return df


def load_features(model: str) -> tuple[np.ndarray, pd.DataFrame]:
    """加载 E1 冻结特征,返回 (features[N,D], ids_df[N])。"""
    d = FEAT_DIR / model
    feat = np.load(d / f"{model}_features.npy")
    ids = pd.read_csv(d / f"{model}_sample_ids.csv")
    return feat, ids


def build_visits_ridirp(df: pd.DataFrame, feat: np.ndarray, ids: pd.DataFrame) -> pd.DataFrame:
    """RIDIRP: 按 (patient_id, series) 聚合成访次。"""
    r = df[df.dataset == "ridirp"].copy()
    r = r.merge(ids, on="sample_id", how="inner")
    r["feat_idx"] = r.index  # ids merge 后行号即特征索引

    # 排除没有 PMA 的访次(无法排序)
    r = r.dropna(subset=["pma"])

    visits = []
    for (pid, series), grp in r.groupby(["patient_id", "series"]):
        grp = grp.sort_values("sample_id")
        idxs = grp["feat_idx"].values
        visit_feat = feat[idxs].mean(axis=0)  # 访次内图像特征均值
        dg = grp["dg"].iloc[0] if "dg" in grp.columns else np.nan
        label = grp["strict_binary_label"].max()
        # 治疗后访视: DG9 或 strict 排除(laser)
        is_post_tx = (
            (dg in POST_TREATMENT_DG)
            or (grp["strict_group"].iloc[0] == "laser/post-treatment")
        )
        visits.append({
            "dataset": "ridirp",
            "patient_id": pid,
            "visit_key": f"ridirp_{pid}_S{int(series)}",
            "time": grp["pma"].iloc[0],  # PMA 周
            "time_type": "pma_weeks",
            "ga": grp["ga"].iloc[0],
            "bw": grp["bw"].iloc[0],
            "sex": grp["sex"].iloc[0],
            "dg": dg,
            "label": int(label) if pd.notna(label) else -1,
            "is_post_treatment": is_post_tx,
            "n_images": len(grp),
            "visit_feat": visit_feat,
        })
    return pd.DataFrame(visits)


def build_visits_ropvl(df: pd.DataFrame, feat: np.ndarray, ids: pd.DataFrame) -> pd.DataFrame:
    """ROP-VL: 按 (patient_id, exam_date) 聚合成访次。"""
    v = df[df.dataset == "rop_vl"].copy()
    v = v.merge(ids, on="sample_id", how="inner")
    v["feat_idx"] = v.index
    v = v.dropna(subset=["exam_date"])
    v["exam_date"] = pd.to_datetime(v["exam_date"])

    visits = []
    for (pid, edate), grp in v.groupby(["patient_id", "exam_date"]):
        grp = grp.sort_values("sample_id")
        idxs = grp["feat_idx"].values
        visit_feat = feat[idxs].mean(axis=0)
        label = grp["strict_binary_label"].max()
        is_post_tx = grp["strict_group"].iloc[0] == "laser/post-treatment"
        visits.append({
            "dataset": "rop_vl",
            "patient_id": pid,
            "visit_key": f"ropvl_{pid}_{edate.strftime('%Y%m%d')}",
            "time": edate,
            "time_type": "exam_date",
            "ga": grp["ga"].iloc[0],
            "bw": grp["bw"].iloc[0],
            "sex": grp["sex"].iloc[0],
            "dg": np.nan,
            "label": int(label) if pd.notna(label) else -1,
            "is_post_treatment": is_post_tx,
            "n_images": len(grp),
            "visit_feat": visit_feat,
        })
    return pd.DataFrame(visits)


def build_longitudinal_samples(
    visits: pd.DataFrame, max_history: int = 5
) -> tuple[pd.DataFrame, dict]:
    """从访次表构建纵向 index 样本。

    每个 index 访次(阴性、非治疗后)对应:
    - 历史访次序列(含当前 index),按时间排序,最多 max_history 次
    - y_next: 下次访视是否阳性
    - y_eventual: 任何后续访视是否阳性
    """
    samples = []
    feat_dim = visits.iloc[0].visit_feat.shape[0]
    sequences = []  # list of (seq_feat[T,D], seq_time[T], seq_mask[T])

    for (ds, pid), grp in visits.groupby(["dataset", "patient_id"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue  # 至少需要 2 次访视才能做纵向预测

        # 时间轴: RIDIRP 用 PMA 周(浮点), ROP-VL 用距首次访视天数
        if grp.time_type.iloc[0] == "pma_weeks":
            times = grp.time.values.astype(float)
        else:
            times = np.array([(t - grp.time.iloc[0]).days for t in grp.time], dtype=float)

        for i in range(n - 1):  # 最后一次访视没有未来,不能作为 index
            row = grp.iloc[i]
            # index 访次必须是阴性且非治疗后
            if row.label != 0 or row.is_post_treatment:
                continue

            future = grp.iloc[i + 1:]
            # 排除治疗后访视对标签的干扰? 不——治疗后阳性仍算事件
            y_next = int(future.iloc[0].label == 1)
            y_eventual = int((future.label == 1).any())

            # 历史序列(含当前 index),最多 max_history 次
            start = max(0, i - max_history + 1)
            hist = grp.iloc[start:i + 1]
            t_seq = len(hist)
            seq_feat = np.zeros((max_history, feat_dim), dtype=np.float32)
            seq_time = np.zeros(max_history, dtype=np.float32)
            seq_mask = np.zeros(max_history, dtype=np.float32)
            for j, (_, hrow) in enumerate(hist.iterrows()):
                seq_feat[j] = hrow.visit_feat
                seq_time[j] = times[start + j]
                seq_mask[j] = 1.0

            # 时间间隔(距 index 访视)
            dt = times[i] - times[start:i + 1]
            seq_dt = np.zeros(max_history, dtype=np.float32)
            seq_dt[:t_seq] = dt[::-1]  # 反转: 最近的在前? 不,保持时间顺序

            seq_idx = len(samples)
            sequences.append({
                "feat": seq_feat,
                "time": seq_time,
                "mask": seq_mask,
                "dt_weeks": seq_dt,
            })

            samples.append({
                "seq_idx": seq_idx,
                "dataset": ds,
                "patient_id": pid,
                "visit_key": row.visit_key,
                "index_time": row.time,
                "ga": row.ga,
                "bw": row.bw,
                "sex": row.sex,
                "n_history": t_seq,
                "n_images_index": row.n_images,
                "y_next": y_next,
                "y_eventual": y_eventual,
                "next_time": future.iloc[0].time,
                "next_label": future.iloc[0].label,
                "n_future_visits": len(future),
            })

    manifest = pd.DataFrame(samples)
    return manifest, sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-model", default="retfound_green",
                    help="E1 特征缓存模型名(默认 retfound_green)")
    ap.add_argument("--max-history", type=int, default=5,
                    help="最大历史访次数(含当前 index)")
    args = ap.parse_args()

    LONG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest v2 ...")
    df = load_manifest()
    print(f"  {len(df)} rows")

    print(f"Loading features: {args.feature_model} ...")
    feat, ids = load_features(args.feature_model)
    print(f"  features {feat.shape}, ids {len(ids)}")

    print("Building visits ...")
    v_ridirp = build_visits_ridirp(df, feat, ids)
    v_ropvl = build_visits_ropvl(df, feat, ids)
    visits = pd.concat([v_ridirp, v_ropvl], ignore_index=True)
    print(f"  RIDIRP visits: {len(v_ridirp)}, ROP-VL visits: {len(v_ropvl)}")
    print(f"  Total visits: {len(visits)}")
    print(f"  Positive visits: {(visits.label==1).sum()}")
    print(f"  Post-treatment visits: {visits.is_post_treatment.sum()}")

    print("Building longitudinal samples ...")
    manifest, sequences = build_longitudinal_samples(visits, args.max_history)
    print(f"  Total index samples: {len(manifest)}")
    print(f"  y_next events: {manifest.y_next.sum()} ({manifest.y_next.mean()*100:.1f}%)")
    print(f"  y_eventual events: {manifest.y_eventual.sum()} ({manifest.y_eventual.mean()*100:.1f}%)")
    print(f"  By dataset:")
    print(manifest.groupby("dataset").agg(
        n=("seq_idx", "count"),
        events_next=("y_next", "sum"),
        events_eventual=("y_eventual", "sum"),
    ).to_string())
    print(f"  Patients: {manifest.patient_id.nunique()}")
    print(f"  History length distribution:")
    print(manifest.n_history.value_counts().sort_index().to_string())

    # Save
    out_manifest = LONG_DIR / f"longitudinal_manifest_{args.feature_model}.csv"
    manifest.to_csv(out_manifest, index=False)
    print(f"\nSaved manifest: {out_manifest}")

    # Save sequences as npz
    seq_feats = np.stack([s["feat"] for s in sequences])
    seq_times = np.stack([s["time"] for s in sequences])
    seq_masks = np.stack([s["mask"] for s in sequences])
    seq_dts = np.stack([s["dt_weeks"] for s in sequences])
    out_seq = LONG_DIR / f"visit_sequences_{args.feature_model}.npz"
    np.savez_compressed(
        out_seq,
        features=seq_feats,
        times=seq_times,
        masks=seq_masks,
        dt_weeks=seq_dts,
    )
    print(f"Saved sequences: {out_seq}")
    print(f"  features: {seq_feats.shape}, times: {seq_times.shape}, masks: {seq_masks.shape}")


if __name__ == "__main__":
    main()
