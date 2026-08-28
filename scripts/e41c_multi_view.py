"""E4.1 消融 C:单视野 → 多视野聚合(RIDIRP 唯一有 visit_id 的公开集)。

协议:复用 E1/E2 冻结探针在 ridirp 折(heldout_dataset=ridirp,subset=test)的逐图预测,
同一患者同一访次的多图预测做均值池化(visit 级),与逐图(image 级)、逐患者(patient 级,
跨访次均值池化)AUROC 对比。假设:筛查场景通常一次访视采集双眼多视野,多视野聚合
应当提升判别稳定性。统计:各单元级 AUROC + 患者级 bootstrap 2000(seed 42)。

输出:outputs/ablation/e41c_multi_view.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.eval.metrics import auroc  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# R3 隔离口径:RIDIRP 折 c* = v1→ep025、heads→final;单权重模型隔离=原协议(数字一致)。
MODELS = ["retfound_green", "dinov2_vits14",
          "ibot_dinov2s_v1", "ibot_dinov2s_heads"]
MODEL_CN = {"retfound_green": "RETFound-Green",
            "dinov2_vits14": "DINOv2-S(起点)",
            "ibot_dinov2s_v1": "NeoROP-FM(iBOT,ep025)",
            "ibot_dinov2s_heads": "NeoROP-FM(+辅助头,final)"}


def pred_path(model: str) -> Path:
    """优先隔离锁定预测(R3),未产出回退旧 probes 目录(单权重同数字)。"""
    iso = REPO / f"outputs/checkpoint_iso/{model}_iso/fold_ridirp_predictions.csv"
    if iso.exists():
        return iso
    legacy = {"ibot_dinov2s_v1": "ibot_dinov2s_v1_ckpt_ep040",
              "ibot_dinov2s_heads": "ibot_dinov2s_heads_ckpt_ep040"}.get(model, model)
    return REPO / f"outputs/probes/{legacy}/fold_ridirp_predictions.csv"


def agg_auc(y: np.ndarray, p: np.ndarray, pids: np.ndarray, seed: int = 42):
    ci = patient_level_bootstrap(y, p, pids, auroc, n_boot=2000, seed=seed)
    return {"auroc": ci["point"], "lo": ci["lower"], "hi": ci["upper"],
            "n_valid_boot": ci["n_valid"]}


def run_model(model: str) -> dict:
    pred = pd.read_csv(pred_path(model))
    te = pred[pred["subset"] == "test"].copy()
    mani = pd.read_csv(REPO / "data/manifests/public_rop_strict_manifest.csv",
                       usecols=["sample_id", "visit_id", "patient_id"]).set_index("sample_id")
    te["visit_id"] = te["sample_id"].map(mani["visit_id"])
    assert te["visit_id"].notna().all(), "RIDIRP 应全部有 visit_id"

    # image 级
    img = agg_auc(te["y"].to_numpy(), te["p"].to_numpy(),
                  te["dataset"] + "|" + te["patient_id"])

    # visit 级:同 (patient, visit) 内均值池化
    gv = te.groupby(["patient_id", "visit_id"], as_index=False).agg(
        p=("p", "mean"), y=("y", "max"), dataset=("dataset", "first"))
    vis = agg_auc(gv["y"].to_numpy(), gv["p"].to_numpy(),
                  gv["dataset"] + "|" + gv["patient_id"])

    # patient 级:跨访次再池化
    gp = te.groupby(["patient_id"], as_index=False).agg(
        p=("p", "mean"), y=("y", "max"), dataset=("dataset", "first"))
    pat = agg_auc(gp["y"].to_numpy(), gp["p"].to_numpy(),
                  gp["dataset"] + "|" + gp["patient_id"])

    return {"model": model, "display": MODEL_CN[model],
            "n_images": len(te), "n_visits": len(gv), "n_patients": len(gp),
            "image_auroc": img["auroc"], "image_lo": img["lo"], "image_hi": img["hi"],
            "visit_auroc": vis["auroc"], "visit_lo": vis["lo"], "visit_hi": vis["hi"],
            "patient_auroc": pat["auroc"], "patient_lo": pat["lo"], "patient_hi": pat["hi"]}


def main() -> None:
    rows = [run_model(m) for m in MODELS]
    df = pd.DataFrame(rows)
    out = REPO / "outputs/ablation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e41c_multi_view.csv", index=False)
    show = df[["display", "n_images", "n_visits", "n_patients",
               "image_auroc", "visit_auroc", "patient_auroc"]].copy()
    for c in ["image_auroc", "visit_auroc", "patient_auroc"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
