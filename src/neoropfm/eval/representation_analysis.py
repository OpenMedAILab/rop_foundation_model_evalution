"""E4.3 表征分析(方案 §21):冻结表征的"发育感知"量化。

仅用 RIDIRP(唯一带 PMA 的公开集;strict-binary 4,330 张,特征已缓存):
1. **PMA 可预测性**:特征 → PMA 的 Ridge 回归 5 折 CV R²。越高 = 表征编码越多的
   发育年龄信息(方案 Q2"新生儿缺什么表征"的定量证据);
2. **DDS(developmental deviation score)**:仅用阴性图像拟合"正常发育轨迹"
   (特征 → PMA Ridge,5 折 CV 外推),DDS = |PMA 预测误差|(单位:周)= 该图像
   偏离正常发育轨迹的周数;检验 DDS 是否预测 strict-binary 阳性
   (AUROC + 患者级 bootstrap CI)——"偏离发育轨迹即患病风险"假设;
3. **PC1–PMA 相关性**:各模型 PCA 第一主方向分数与 PMA 的 |Spearman ρ|,
   表征发育轨迹的全局有序性(§21 正常发育轨迹有序性检验);
4. **PCA 前 2 维坐标 / 逐图像 DDS**:存盘供报告阶段绘图。

输出:outputs/representation/{pma_r2_dds.csv, pca_coords_{model}.npy, dds_scores_{model}.npy}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import load_yaml, seed_everything  # noqa: E402
from neoropfm.eval.metrics import auroc  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402


def load_feats_any(model: str, cfg: dict):
    """extract 目录优先,yesterday 缓存回退(与 label_efficiency 同逻辑)。"""
    from neoropfm.train.probe import load_features
    try:
        return load_features(cfg["feature_source"], model,
                             Path(cfg["cache_dir"]), Path(cfg["extract_dir"]))
    except FileNotFoundError:
        other = "cache" if cfg["feature_source"] == "extract" else "extract"
        return load_features(other, model, Path(cfg["cache_dir"]), Path(cfg["extract_dir"]))


def ridirp_slice(feat: np.ndarray, ids: list[str], ssl_manifest: Path, strict_manifest: Path):
    """RIDIRP strict-binary 子集 + PMA + label + patient id。"""
    ssl = pd.read_csv(ssl_manifest).set_index("sample_id")["pma"]
    strict = pd.read_csv(strict_manifest).set_index("sample_id")
    keep = [i for i, s in enumerate(ids)
            if strict.loc[s, "dataset"] == "ridirp" and strict.loc[s, "include_strict_binary"]]
    keep = np.array(keep)
    sub_ids = [ids[i] for i in keep]
    pma = ssl.loc[sub_ids].to_numpy()
    labels = strict.loc[sub_ids, "strict_binary_label"].to_numpy()
    pids = (strict.loc[sub_ids, "dataset"].astype(str) + "|"
            + strict.loc[sub_ids, "patient_id"].astype(str)).to_numpy()
    return feat[keep], pma, labels, pids, sub_ids


def pma_r2(X: np.ndarray, pma: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=seed)
    r2s = []
    for itr, ite in kf.split(Xs):
        reg = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Xs[itr], pma[itr])
        r2s.append(r2_score(pma[ite], reg.predict(Xs[ite])))
    return float(np.mean(r2s))


def dds_scores(X: np.ndarray, pma: np.ndarray, labels: np.ndarray,
               seed: int) -> np.ndarray:
    """阴性拟合轨迹、全样本外推,DDS = |PMA 预测误差|(周)。"""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=seed)
    dds = np.zeros(len(X))
    for itr, ite in kf.split(Xs):
        mask = labels[itr] == 0
        reg = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Xs[itr][mask], pma[itr][mask])
        dds[ite] = np.abs(pma[ite] - reg.predict(Xs[ite]))
    return dds


def run_model(model: str, cfg: dict) -> dict:
    feat, ids = load_feats_any(model, cfg)
    X, pma, labels, pids, _ = ridirp_slice(
        feat, ids, Path(cfg["ssl_manifest"]), Path(cfg["strict_manifest"]))
    seed = cfg.get("seed", 0)
    r2 = pma_r2(X, pma, seed)
    dds = dds_scores(X, pma, labels, seed)

    ci = patient_level_bootstrap(labels, dds, pids, auroc,
                                 n_boot=cfg.get("n_boot", 2000), seed=seed)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit(Xs)
    scores1 = pca.components_[0] @ Xs.T
    rho = float(np.abs(pd.Series(scores1).corr(pd.Series(pma), method="spearman")))
    coords = pca.transform(Xs)[:, :2]
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"pca_coords_{model}.npy",
            np.column_stack([coords[:, 0], coords[:, 1], pma, labels]))
    np.save(out / f"dds_scores_{model}.npy", np.column_stack([dds, labels]))

    row = {"model": model, "pma_r2_cv": r2, "pc1_pma_spearman": rho,
           "dds_auroc": ci["point"], "dds_auroc_lo": ci["lower"],
           "dds_auroc_hi": ci["upper"], "n_valid_boot": ci["n_valid"],
           "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum())}
    print(f"[{model}] PMA R²={r2:.4f}  |PC1–PMA ρ|={rho:.4f}  "
          f"DDS AUROC={ci['point']:.4f} [{ci['lower']:.3f}, {ci['upper']:.3f}]", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 0))
    models = [args.model] if args.model else cfg["models"]
    rows = [run_model(m, cfg) for m in models]
    df = pd.DataFrame(rows).sort_values("pma_r2_cv", ascending=False)
    df.to_csv(Path(cfg["output_dir"]) / "pma_r2_dds.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
