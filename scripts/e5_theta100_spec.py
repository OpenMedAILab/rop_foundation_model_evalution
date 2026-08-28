"""θ100 特异度聚合(阈值锁定部署参考)。

锁定版 Table S5 将 θ100 特异度标 "undefined(阈值=最极端阳性训练分数)"。逐折
CSV(outputs/e5_locked/{model}_e5_locked.csv, op=theta100)中其实已计算
test_spec(阈值为最小阳性训练分数,阴性分数低于该阈值即 TN,特异度可算)。
本脚本只做聚合(访视级,与锁定 E5 主口径一致),并额外给出折最大值。

输出(新文件,不覆盖锁定产物):outputs/e5_locked/all_models_theta100_spec.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/e5_locked"


def main() -> None:
    rows = []
    for f in sorted(OUT.glob("*_e5_locked.csv")):
        if f.name.startswith("all_models"):
            continue
        df = pd.read_csv(f)
        sub = df[(df["op"] == "theta100") & (df["level"] == "visit")]
        if not len(sub):
            continue
        rows.append({"model": sub["model"].iloc[0], "n_folds": len(sub),
                     "mean_test_spec": round(sub["test_spec"].mean(), 4),
                     "max_test_spec": round(sub["test_spec"].max(), 4),
                     "mean_test_sens": round(sub["test_sens"].mean(), 4)})
    agg = pd.DataFrame(rows).sort_values("mean_test_spec", ascending=False)
    agg.to_csv(OUT / "all_models_theta100_spec.csv", index=False)
    print("saved →", OUT / "all_models_theta100_spec.csv")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
