"""冻结特征提取(E1 新模型 / E2 SSL 继续预训练后的新编码器)。

流程:manifest 图像 → backbone 预处理 → 冻结前向 → 缓存到
outputs/features/{model}/{model}_features.npy + _sample_ids.csv + _feature_meta.json。
probe 阶段(probe.py, feature_source=extract)与统计阶段零重算。

GPU 排队:experiments/gpu_queue.sh 等待显存空闲后调用本脚本。
运行:
  python -m neoropfm.train.extract_features --config configs/extract.yaml --model dinov2_vitb14
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V1, gpu_free_mb, load_yaml, seed_everything  # noqa: E402
from neoropfm.models.backbones import get_backbone, get_ssl_backbone  # noqa: E402


class ManifestImages(Dataset):
    """按 manifest 顺序读取图像(仅 include_strict_binary==1)。"""

    def __init__(self, manifest: pd.DataFrame, transform):
        self.paths = manifest["image_path"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def run(model_key: str, cfg: dict, limit: int | None = None,
        ckpt: Path | None = None, base_key: str | None = None,
        manifest_path: Path | None = None, strict_filter: bool = True) -> None:
    # E2.5:--ckpt/--base-key 时挂 E2 SSL 继续预训练权重(提取口径与基座完全一致)
    spec = get_ssl_backbone(model_key, ckpt, base_key) if ckpt else get_backbone(model_key)
    seed_everything(cfg.get("seed", 0))
    if cfg.get("cpu_threads"):
        torch.set_num_threads(cfg["cpu_threads"])  # CPU 提取时限核,共享机器保持礼貌
    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config 要求 GPU 但当前不可用;检查 gpu_queue.sh 或改 device=cpu")

    min_free = cfg.get("check_gpu_free_mb", 0)
    if device.startswith("cuda") and min_free:
        ok, free = gpu_free_mb(min_free)
        if not ok:
            raise RuntimeError(f"GPU 空闲显存 {free}MB < 要求 {min_free}MB;由 gpu_queue.sh 排队重试")

    manifest = pd.read_csv(manifest_path if manifest_path is not None else MANIFEST_V1)
    if strict_filter and "include_strict_binary" in manifest.columns:
        manifest = manifest[manifest["include_strict_binary"] == 1].reset_index(drop=True)
    if limit:
        manifest = manifest.head(limit)
    print(f"[{model_key}] {len(manifest)} images | device={device} | {spec.weights_note}")

    loader = DataLoader(
        # 注意:传 spec.compose(裸 Compose)而非 spec.transform(绑定方法)——后者会把
        # 整个 BackboneSpec 带入 multiprocessing pickle,spawn/forkserver 下失败
        ManifestImages(manifest, spec.compose),
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 4),
        shuffle=False,
        drop_last=False,
    )
    model = spec.build(device)
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            feats.append(spec.extract(model, batch.to(device)))
    feats = np.concatenate(feats, axis=0)
    elapsed = time.time() - t0

    out_dir = Path(cfg["output_dir"]) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{model_key}_features.npy", feats)
    manifest[["sample_id"]].to_csv(out_dir / f"{model_key}_sample_ids.csv", index=False)
    meta = {
        "model": model_key,
        "image_size": spec.input_size,
        "n_samples": int(len(feats)),
        "feature_dim": int(feats.shape[1]),
        "device": device,
        "elapsed_sec": round(elapsed, 2),
        "preprocessing": spec.preprocessing,
        "weights_note": spec.weights_note,
    }
    (out_dir / f"{model_key}_feature_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {out_dir}/{model_key}_features.npy {feats.shape} ({elapsed:.1f}s)")
    print(json.dumps(meta, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 张图(冒烟测试)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="E2 SSL checkpoint(pth);与 --base-key 连用,替换基座权重")
    ap.add_argument("--base-key", default=None, help="SSL 起点 backbone 键(如 dinov2_vits14)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="覆盖提取 manifest(FARFUM 全量/外部折合并清单用)")
    ap.add_argument("--no-strict-filter", action="store_true",
                    help="不做 include_strict_binary==1 过滤(全量提取用)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    run(args.model, cfg, limit=args.limit, ckpt=args.ckpt, base_key=args.base_key,
        manifest_path=args.manifest, strict_filter=not args.no_strict_filter)


if __name__ == "__main__":
    main()
