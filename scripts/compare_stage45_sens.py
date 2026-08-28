"""标签规则近似敏感性对比:v1 规则 vs 稿件近似规则(A-ROP/stage4-5)逐折 AUROC。

输出:outputs/E1_label_rule_sensitivity.md 所需的表格数据。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

MODELS = ["retfound_green", "retfound_mae_cfp", "dinov2_vits14", "dinov2_vitb14",
          "convnext_tiny", "efficientnet_b0"]
FOLDS = ["farfum_rop", "ridirp", "rop_vl", "szeh_irops"]


def load(base: Path, model: str) -> pd.DataFrame:
    f = base / model / f"{model}_probe_lodo_metrics.csv"
    return pd.read_csv(f).set_index("heldout_dataset")


rows = []
for model in MODELS:
    v1 = load(Path("outputs/probes"), model)
    va = load(Path("outputs/probes_stage45_sens"), model)
    for fold in FOLDS:
        rows.append({
            "model": model, "fold": fold,
            "v1": v1.loc[fold, "auroc"], "variant": va.loc[fold, "auroc"],
            "diff": va.loc[fold, "auroc"] - v1.loc[fold, "auroc"],
        })
df = pd.DataFrame(rows)
df.to_csv("outputs/aggregate_stage45_sens_fold_diffs.csv", index=False)

print("逐折 AUROC(v1 → 稿件近似)diff:")
piv = df.pivot_table(index="model", columns="fold", values="diff").round(4)
piv["mean"] = df.groupby("model")["diff"].mean().round(4)
print(piv.to_string())
print("\n排名变化(mean AUROC):")
for name, base in [("v1", "outputs/probes"), ("variant", "outputs/probes_stage45_sens")]:
    means = {m: load(Path(base), m)["auroc"].mean() for m in MODELS}
    rank = sorted(means.items(), key=lambda kv: -kv[1])
    print(f"  {name}: " + " > ".join(f"{m}({v:.3f})" for m, v in rank))
