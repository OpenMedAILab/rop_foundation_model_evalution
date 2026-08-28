"""Temporal Transformer 纵向预测模型(E3.3 + E3.4)。

轻量级时间 Transformer,处理访次特征序列 + 时间编码 + 临床变量,
输出 next-visit 阳性概率。考虑到样本量极小(252 样本/19 事件),
模型刻意保持小参数量 + 强正则化。

E3.3: Temporal Transformer
E3.4: 深度集成(5 个不同种子的模型)+ 三态分流
      (treat / observe / reschedule,基于预测概率和不确定性)

运行:
  python -m neoropfm.train.temporal_transformer
  python -m neoropfm.train.temporal_transformer --feature-model retfound_green
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import OUTPUTS_DIR, seed_everything  # noqa: E402
from neoropfm.eval.metrics import compute_all_metrics, threshold_at_sensitivity  # noqa: E402

LONG_DIR = OUTPUTS_DIR / "longitudinal"


class TimeEncoding(nn.Module):
    """连续时间编码(正弦位置编码的连续版本)。"""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        # 可学习的频率
        self.w = nn.Parameter(torch.randn(dim // 2) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B, T] → [B, T, dim]"""
        proj = t.unsqueeze(-1) * self.w.unsqueeze(0).unsqueeze(0) * self.scale
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TemporalTransformer(nn.Module):
    """访次级 Temporal Transformer。

    输入:
      seq_feat  [B, T, D]  访次冻结特征
      seq_mask  [B, T]     有效访次掩码
      seq_dt    [B, T]     距 index 访视的时间间隔(周)
      clinical  [B, C]     临床变量(GA/BW/sex/PMA/n_history)
    输出:
      risk      [B]        next-visit 阳性概率
    """

    def __init__(
        self,
        feat_dim: int,
        clinical_dim: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_enc = TimeEncoding(d_model, scale=0.1)
        self.clin_proj = nn.Sequential(
            nn.Linear(clinical_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2 + clinical_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, seq_feat, seq_mask, seq_dt, clinical):
        B, T, D = seq_feat.shape

        # 特征投影 + 时间编码
        x = self.feat_proj(seq_feat)
        t_enc = self.time_enc(seq_dt)
        x = x + t_enc

        # Transformer(padding mask: True = 忽略)
        pad_mask = ~seq_mask.bool()
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # 聚合: 最后一个有效访次(index) + 全局均值池化
        # 找到每个样本最后一个有效位置
        last_idx = seq_mask.long().sum(dim=1) - 1  # [B]
        last_token = x[torch.arange(B), last_idx]  # [B, d_model]

        # 掩码均值池化
        mask_exp = seq_mask.unsqueeze(-1)
        mean_token = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        # 拼接
        combined = torch.cat([last_token, mean_token, clinical], dim=-1)
        logit = self.risk_head(combined).squeeze(-1)
        return logit


def encode_sex(sex_val) -> float:
    if pd.isna(sex_val):
        return 0.5
    s = str(sex_val).lower().strip()
    if s in ("boy", "male", "m", "1"):
        return 1.0
    if s in ("girl", "female", "f", "0"):
        return 0.0
    return 0.5


def prepare_tensors(manifest, seq_feat, seq_mask, seq_dt, scaler=None, fit=False):
    """准备 PyTorch 张量。"""
    # 临床变量
    C = np.zeros((len(manifest), 5), dtype=np.float32)
    for i, (_, row) in enumerate(manifest.iterrows()):
        C[i, 0] = row.ga if pd.notna(row.ga) else np.nan
        C[i, 1] = row.bw if pd.notna(row.bw) else np.nan
        C[i, 2] = encode_sex(row.sex)
        # RIDIRP index_time 是 PMA 周(浮点); ROP-VL 是日期字符串,稍后用 seq_dt 填充
        if row.dataset == "ridirp":
            try:
                C[i, 3] = float(row.index_time)
            except (ValueError, TypeError):
                C[i, 3] = np.nan
        else:
            C[i, 3] = np.nan
        C[i, 4] = row.n_history

    # ROP-VL 的时间用 seq_dt 的最后一个有效值
    for i, row in manifest.iterrows():
        if row.dataset == "rop_vl":
            m = seq_mask[i].astype(bool)
            C[i, 3] = seq_dt[i, m][-1]  # 距首次访视周数

    # 填充缺失值
    col_means = np.nanmean(C, axis=0)
    for j in range(C.shape[1]):
        mask = np.isnan(C[:, j])
        C[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    if fit:
        scaler = StandardScaler().fit(C)
    C = scaler.transform(C)

    X = torch.FloatTensor(seq_feat)
    M = torch.FloatTensor(seq_mask)
    D = torch.FloatTensor(seq_dt)
    C = torch.FloatTensor(C)
    y = torch.FloatTensor(manifest.y_next.values)

    return X, M, D, C, y, scaler


def train_fold(
    model, Xtr, Mtr, Dtr, Ctr, ytr,
    Xva, Mva, Dva, Cva, yva,
    pos_weight=None, epochs=200, lr=1e-3, wd=1e-2, patience=30, device="cpu",
):
    """训练单个 fold,返回最佳模型的验证预测。"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    Xtr, Mtr, Dtr, Ctr, ytr = [t.to(device) for t in [Xtr, Mtr, Dtr, Ctr, ytr]]
    Xva, Mva, Dva, Cva, yva = [t.to(device) for t in [Xva, Mva, Dva, Cva, yva]]

    best_auprc = -1
    best_state = None
    wait = 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logit = model(Xtr, Mtr, Dtr, Ctr)
        loss = criterion(logit, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(Xva, Mva, Dva, Cva)).cpu().numpy()
        from sklearn.metrics import average_precision_score
        if len(np.unique(yva.cpu().numpy())) == 2:
            auprc = average_precision_score(yva.cpu().numpy(), pva)
        else:
            auprc = 0.0

        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva, Mva, Dva, Cva)).cpu().numpy()
        ptr = torch.sigmoid(model(Xtr, Mtr, Dtr, Ctr)).cpu().numpy()
    return model.cpu(), pva, ptr


def run_temporal_transformer(
    manifest, seq_feat, seq_mask, seq_dt,
    n_splits=5, seed=0, n_ensemble=5, device="cpu",
):
    """运行 Temporal Transformer + 深度集成。"""
    y = manifest.y_next.values.astype(int)
    groups = manifest.patient_id.values
    D_feat = seq_feat.shape[2]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 每个 fold 的集成预测
    fold_preds_ensemble = []  # list of [n_ensemble, n_test]
    fold_y = []
    fold_groups = []
    fold_train_preds = []  # for threshold locking

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(seq_feat, y, groups)):
        print(f"  Fold {fold+1}/{n_splits}: train={len(tr_idx)} (pos={y[tr_idx].sum()}), "
              f"test={len(te_idx)} (pos={y[te_idx].sum()})")

        # 准备数据
        man_tr = manifest.iloc[tr_idx].reset_index(drop=True)
        man_te = manifest.iloc[te_idx].reset_index(drop=True)
        Xtr, Mtr, Dtr, Ctr, ytr, scaler = prepare_tensors(
            man_tr, seq_feat[tr_idx], seq_mask[tr_idx], seq_dt[tr_idx], fit=True
        )
        Xte, Mte, Dte, Cte, yte, _ = prepare_tensors(
            man_te, seq_feat[te_idx], seq_mask[te_idx], seq_dt[te_idx],
            scaler=scaler, fit=False,
        )

        # 处理 ROP-VL index_time (NaN) - prepare_tensors 已处理

        # 类别权重
        n_pos = ytr.sum()
        n_neg = len(ytr) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

        # 集成
        ensemble_p = []
        ensemble_ptr = []
        for ens in range(n_ensemble):
            seed_everything(seed + ens * 100 + fold)
            model = TemporalTransformer(
                feat_dim=D_feat, d_model=128, n_heads=4, n_layers=2, dropout=0.3,
            )
            model, pte, ptr = train_fold(
                model, Xtr, Mtr, Dtr, Ctr, ytr,
                Xte, Mte, Dte, Cte, yte,
                pos_weight=pos_weight, epochs=200, lr=1e-3, wd=1e-2,
                patience=30, device=device,
            )
            ensemble_p.append(pte)
            ensemble_ptr.append(ptr)

        fold_preds_ensemble.append(np.array(ensemble_p))  # [n_ens, n_test]
        fold_y.append(yte.numpy().astype(int))
        fold_groups.append(groups[te_idx])
        fold_train_preds.append(np.array(ensemble_ptr))

    return fold_preds_ensemble, fold_y, fold_groups, fold_train_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-model", default="retfound_green")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    LONG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(LONG_DIR / f"longitudinal_manifest_{args.feature_model}.csv")
    seq = np.load(LONG_DIR / f"visit_sequences_{args.feature_model}.npz")
    seq_feat = seq["features"].astype(np.float32)
    seq_mask = seq["masks"].astype(np.float32)
    seq_dt = seq["dt_weeks"].astype(np.float32)

    print(f"Loaded {len(manifest)} samples, features {seq_feat.shape}")
    print(f"Events: {manifest.y_next.sum()} / {len(manifest)}")
    print(f"Device: {args.device}")

    # 运行 Temporal Transformer + 集成
    print(f"\n=== Temporal Transformer (ensemble={args.n_ensemble}) ===")
    fold_ens, fold_y, fold_groups, fold_train_ens = run_temporal_transformer(
        manifest, seq_feat, seq_mask, seq_dt,
        n_splits=args.n_splits, seed=args.seed,
        n_ensemble=args.n_ensemble, device=args.device,
    )

    # 汇总结果
    all_y = np.concatenate(fold_y)
    all_groups = np.concatenate(fold_groups)
    all_p_mean = np.concatenate([p.mean(axis=0) for p in fold_ens])
    all_p_std = np.concatenate([p.std(axis=0) for p in fold_ens])

    # 训练折预测(用于阈值锁定)
    all_ptr_mean = np.concatenate([p.mean(axis=0) for p in fold_train_ens])
    all_ytr = np.concatenate([
        manifest.iloc[tr_idx].y_next.values
        for tr_idx, _ in StratifiedGroupKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        ).split(seq_feat, manifest.y_next.values, manifest.patient_id.values)
    ])

    thresholds = {
        "sens95": threshold_at_sensitivity(all_ytr, all_ptr_mean, 0.95),
        "sens98": threshold_at_sensitivity(all_ytr, all_ptr_mean, 0.98),
    }
    m = compute_all_metrics(all_y, all_p_mean, thresholds=thresholds)

    print(f"\n=== Results ===")
    print(f"AUROC: {m['auroc']:.4f}")
    print(f"AUPRC: {m['auprc']:.4f}")
    print(f"spec@95sens: {m['spec@sens95']:.4f}")
    print(f"spec@98sens: {m['spec@sens98']:.4f}")
    print(f"Brier: {m['brier']:.4f}")
    print(f"ECE: {m['ece']:.4f}")

    # E3.4 三态分流
    # treat: p >= 0.5, observe: 0.2 <= p < 0.5, reschedule: p < 0.2
    # 同时考虑不确定性(ensemble std)
    n_treat = int((all_p_mean >= 0.5).sum())
    n_observe = int(((all_p_mean >= 0.2) & (all_p_mean < 0.5)).sum())
    n_reschedule = int((all_p_mean < 0.2).sum())
    # 高不确定性样本
    high_unc = all_p_std > np.median(all_p_std) + all_p_std.std()
    print(f"\nTriaging: treat={n_treat}, observe={n_observe}, reschedule={n_reschedule}")
    print(f"High uncertainty: {high_unc.sum()} samples")

    # 保存
    results = {
        "model": "temporal_transformer",
        "feature_model": args.feature_model,
        "n_samples": len(all_y),
        "n_events": int(all_y.sum()),
        "n_ensemble": args.n_ensemble,
        **m,
        "n_treat": n_treat,
        "n_observe": n_observe,
        "n_reschedule": n_reschedule,
        "n_high_uncertainty": int(high_unc.sum()),
    }
    pd.DataFrame([results]).to_csv(
        LONG_DIR / f"temporal_transformer_metrics_{args.feature_model}.csv", index=False
    )

    # 保存预测
    preds_df = pd.DataFrame({
        "patient_id": all_groups,
        "y": all_y,
        "p_mean": all_p_mean,
        "p_std": all_p_std,
        "triage": np.where(all_p_mean >= 0.5, "treat",
                  np.where(all_p_mean >= 0.2, "observe", "reschedule")),
    })
    preds_df.to_csv(
        LONG_DIR / f"temporal_transformer_predictions_{args.feature_model}.csv",
        index=False,
    )
    print(f"\nSaved metrics and predictions to {LONG_DIR}")


if __name__ == "__main__":
    main()
