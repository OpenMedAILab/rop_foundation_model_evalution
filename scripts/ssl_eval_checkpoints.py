"""E2.5 SSL checkpoint 选择与 leave-test-out 敏感度(E2 训练后运行)。

对 outputs/ssl/{run}/ 下全部 ckpt_ep*.pth(含 ckpt_final.pth):
1. 特征提取:extract_features --ckpt/--base-key(提取口径与基座完全一致);
2. frozen probe:与 E1 主基准同协议(StandardScaler + balanced LR,4 折 LODO);
3. 按 4 折 mean AUROC 选最优 epoch,产出选择表 + best_ckpt 副本。

之后 probe 评估即可直接用最优 checkpoint 作为 NeoROP-FM 编码器,与
E1 六模型同表对比(主表口径一致,仅编码器权重不同)。

运行:
  python3 scripts/ssl_eval_checkpoints.py --run-dir outputs/ssl/ibot_dinov2s_v1 \
      --base-key dinov2_vits14 \
      --extract-config configs/extract.yaml \
      --probe-config configs/probe_extract.yaml
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def extract_and_probe(ckpt: Path, base_key: str, extract_cfg: str, probe_cfg: str,
                      model_key: str) -> float:
    """提取 + probe 一个 checkpoint,返回 4 折 mean AUROC。"""
    extract_dir = None
    import yaml as _yaml
    with open(REPO / extract_cfg, encoding="utf-8") as f:
        extract_dir = _yaml.safe_load(f)["output_dir"]
    with open(REPO / probe_cfg, encoding="utf-8") as f:
        probe_dir = _yaml.safe_load(f)["output_dir"]

    feats = REPO / extract_dir / model_key / f"{model_key}_features.npy"
    metrics = REPO / probe_dir / model_key / f"{model_key}_probe_lodo_metrics.csv"
    if not metrics.exists():
        if not feats.exists():
            print(f"  [extract] {model_key}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "neoropfm.train.extract_features",
                 "--config", extract_cfg, "--model", model_key,
                 "--ckpt", str(ckpt), "--base-key", base_key],
                cwd=REPO, check=True,
            )
        print(f"  [probe] {model_key}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "neoropfm.train.probe",
             "--config", probe_cfg, "--model", model_key],
            cwd=REPO, check=True,
        )
    m = pd.read_csv(metrics)
    return float(m["auroc"].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/ssl/{run}")
    ap.add_argument("--base-key", required=True, help="起点 backbone(dinov2_vits14 / retfound_green)")
    ap.add_argument("--extract-config", default="configs/extract.yaml")
    ap.add_argument("--probe-config", default="configs/probe_extract.yaml")
    args = ap.parse_args()

    run_dir = REPO / args.run_dir
    ckpts = sorted(run_dir.glob("ckpt_ep*.pth")) + sorted(run_dir.glob("ckpt_final.pth"))
    if not ckpts:
        raise FileNotFoundError(f"{run_dir} 下没有 ckpt_*.pth")

    rows = []
    for ckpt in ckpts:
        key = f"{run_dir.name}_{ckpt.stem}"
        print(f"== {key} ==", flush=True)
        mean_auc = extract_and_probe(ckpt, args.base_key, args.extract_config,
                                     args.probe_config, key)
        rows.append({"checkpoint": ckpt.name, "model_key": key, "mean_auroc": mean_auc})
        print(f"  mean_auroc={mean_auc:.4f}", flush=True)

    table = pd.DataFrame(rows).sort_values("mean_auroc", ascending=False)
    table.to_csv(run_dir / "checkpoint_selection.csv", index=False)
    best = table.iloc[0]
    shutil.copy(run_dir / best["checkpoint"], run_dir / "best_ckpt.pth")
    print(f"\n最优:{best['checkpoint']} mean_auroc={best['mean_auroc']:.4f} → best_ckpt.pth")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
