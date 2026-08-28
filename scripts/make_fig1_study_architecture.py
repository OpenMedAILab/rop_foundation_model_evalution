"""Generate the integrated study-architecture overview for manuscript Figure 1.

The figure is authored at its final 6.5-inch manuscript width.  SVG is retained as
the editable master and a 600-dpi PNG is emitted for Word compatibility.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#1E5068"
BLUE = "#3A7592"
TEAL = "#287F79"
ORANGE = "#C86E35"
RED = "#B84D50"
INK = "#1F2933"
MUTED = "#5A6872"
LINE = "#C8D3D9"


def generate(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.fonttype"] = "none"

    fig = plt.figure(figsize=(6.5, 4.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    def rounded(x, y, w, h, fc="white", ec=LINE, lw=0.75, radius=0.09, z=1,
                linestyle="-"):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.015,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=linestyle,
            zorder=z,
        )
        ax.add_patch(p)
        return p

    def section(letter, title, y, h, accent, fill):
        rounded(0.08, y, 12.84, h, fc=fill, ec=LINE, lw=0.75, radius=0.12, z=0)
        ax.add_patch(Rectangle((0.08, y), 0.09, h, facecolor=accent,
                               edgecolor="none", zorder=1))
        rounded(0.24, y + h - 0.42, 0.38, 0.29, fc="white", ec=INK,
                lw=0.75, radius=0.025, z=2)
        ax.text(0.43, y + h - 0.275, letter, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=INK, zorder=3)
        ax.text(0.76, y + h - 0.27, title, ha="left", va="center",
                fontsize=8.2, fontweight="bold", color=accent, zorder=3)

    def card(x, y, w, h, title, accent, body=None, fc="white",
             title_fs=6.25, body_fs=5.35, header_h=0.29):
        rounded(x, y, w, h, fc=fc, ec=accent, lw=0.85, radius=0.075, z=2)
        ax.add_patch(Rectangle((x + 0.018, y + h - header_h), w - 0.036,
                               header_h - 0.018, facecolor=accent,
                               edgecolor="none", zorder=3))
        ax.text(x + w / 2, y + h - header_h / 2 - 0.01, title,
                ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color="white", zorder=4)
        if body:
            ax.text(x + w / 2, y + (h - header_h) / 2, body,
                    ha="center", va="center", fontsize=body_fs,
                    color=INK, linespacing=1.17, zorder=4)

    def pill(x, y, w, h, label, detail, accent, fill, label_fs=5.25,
             detail_fs=4.85):
        rounded(x, y, w, h, fc=fill, ec=accent, lw=0.65,
                radius=0.045, z=3)
        ax.text(x + 0.13, y + h / 2, label, ha="left", va="center",
                fontsize=label_fs, fontweight="bold", color=accent, zorder=4)
        ax.text(x + w - 0.13, y + h / 2, detail, ha="right", va="center",
                fontsize=detail_fs, color=INK, zorder=4)

    def arrow(x1, y1, x2, y2, label=None, label_dx=0, label_dy=0.11,
              color=INK, lw=1.0, style="-|>", connection=None, z=5):
        p = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8,
            linewidth=lw, color=color, shrinkA=1.4, shrinkB=1.4,
            connectionstyle=connection, zorder=z,
        )
        ax.add_patch(p)
        if label:
            ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy,
                    label, ha="center", va="bottom", fontsize=4.75,
                    color=MUTED, zorder=z + 1,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="none", alpha=0.95))

    # ------------------------------------------------------------------
    # A. Data ecosystem: expose the overlap that motivates source isolation.
    # ------------------------------------------------------------------
    section("A", "Data ecosystem and source separation", 7.54, 1.58,
            NAVY, "#F3F7F9")
    card(0.50, 7.72, 4.03, 0.88, "Labeled benchmark  |  7,815 images", NAVY,
         "FARFUM-RoP  ·  RIDIRP  ·  ROP-VL  ·  SZEH-iROPS\n"
         "TR-ROP proxy: 1,765 positive (22.6%)",
         title_fs=5.95, body_fs=4.85)
    card(4.82, 7.72, 3.78, 0.88, "Unlabeled infant corpus  |  10,656 images", TEAL,
         "Same four sources  ·  labels withheld\nLicense-clean continued-pretraining pool",
         title_fs=5.65, body_fs=4.85, fc="#FCFEFD")
    card(9.17, 7.72, 3.30, 0.88, "Untouched external cohort", ORANGE,
         "HVDROPDB  ·  185 images  ·  CC BY 4.0\nExcluded from all model development",
         title_fs=5.9, body_fs=4.75, fc="#FFFDFC")
    arrow(4.53, 8.15, 4.82, 8.15, "source overlap", label_dy=0.09,
          color=TEAL, lw=0.9, style="<->")
    ax.plot([8.88, 8.88], [7.68, 8.64], color=RED, lw=1.0,
            linestyle=(0, (3, 2)), zorder=3)
    # Minimal lock glyph: separation, not a decorative icon.
    rounded(8.75, 8.12, 0.26, 0.22, fc="white", ec=RED, lw=0.8,
            radius=0.025, z=4)
    ax.add_patch(FancyBboxPatch((8.79, 8.29), 0.18, 0.19,
                               boxstyle="round,pad=0,rounding_size=0.07",
                               facecolor="none", edgecolor=RED,
                               linewidth=0.8, zorder=4))

    # ------------------------------------------------------------------
    # B. Model architecture: starts, adaptation routes, and common probe.
    # ------------------------------------------------------------------
    section("B", "Model comparison and representation pipeline", 4.56, 2.72,
            TEAL, "#F2F8F7")
    card(0.50, 4.79, 3.25, 1.93, "Starting encoders  |  6 baselines", NAVY,
         title_fs=5.95)
    rounded(0.72, 5.73, 2.81, 0.55, fc="#EAF2F6", ec=BLUE,
            lw=0.65, radius=0.045, z=3)
    ax.text(0.86, 6.11, "General vision", ha="left", va="center",
            fontsize=4.95, fontweight="bold", color=BLUE, zorder=4)
    ax.text(2.13, 5.87, "DINOv2-S/B  ·  ConvNeXt-Tiny\nEfficientNet-B0",
            ha="center", va="center", fontsize=3.95, color=INK,
            linespacing=1.0, zorder=4)
    rounded(0.72, 5.05, 2.81, 0.55, fc="#FAEEE7", ec=ORANGE,
            lw=0.65, radius=0.045, z=3)
    ax.text(0.86, 5.43, "Adult retinal", ha="left", va="center",
            fontsize=4.95, fontweight="bold", color=ORANGE, zorder=4)
    ax.text(2.13, 5.20, "RETFound-Green  ·  RETFound-MAE",
            ha="center", va="center", fontsize=4.15, color=INK, zorder=4)
    ax.text(2.13, 4.92, "Original weights retained as unadapted comparators",
            ha="center", va="center", fontsize=4.55, color=MUTED, zorder=4)

    card(4.11, 4.79, 4.52, 1.93, "Infant-domain continued pretraining  |  3 routes", TEAL,
         title_fs=5.8)
    ax.text(6.37, 6.28, "Unlabeled corpus  ·  no task labels  ·  isolated checkpoint selection",
            ha="center", va="center", fontsize=4.55, color=MUTED, zorder=4)
    pill(4.34, 5.75, 4.06, 0.34, "DINOv2-S → iBOT-CP",
         "pure SSL", TEAL, "#E8F4F2", label_fs=4.9, detail_fs=4.65)
    pill(4.34, 5.30, 4.06, 0.34, "DINOv2-S → iBOT-CP",
         "PMA + visit-consistency heads", NAVY, "#EAF1F5",
         label_fs=4.8, detail_fs=4.4)
    pill(4.34, 4.85, 4.06, 0.34, "RETFound-Green → MAE-CP",
         "reconstruction control", ORANGE, "#FAEEE7",
         label_fs=4.75, detail_fs=4.4)

    card(8.98, 4.79, 3.49, 1.93, "Common frozen evaluation", NAVY,
         title_fs=6.0)
    rounded(9.22, 6.03, 3.01, 0.34, fc="#EAF2F6", ec=BLUE,
            lw=0.65, radius=0.045, z=3)
    ax.text(10.725, 6.20, "6 unadapted + 3 adapted = 9 encoders",
            ha="center", va="center", fontsize=4.75, fontweight="bold",
            color=NAVY, zorder=4)
    for yy, text in [(5.61, "Frozen image features"),
                     (5.20, "StandardScaler"),
                     (4.83, "Class-balanced L2 linear probe")]:
        rounded(9.46, yy, 2.53, 0.28, fc="white", ec=NAVY,
                lw=0.6, radius=0.04, z=3)
        ax.text(10.725, yy + 0.14, text, ha="center", va="center",
                fontsize=4.75, color=INK, zorder=4)
    arrow(10.725, 6.03, 10.725, 5.89, None, lw=0.7)
    arrow(10.725, 5.61, 10.725, 5.48, None, lw=0.7)
    arrow(10.725, 5.20, 10.725, 5.11, None, lw=0.7)
    arrow(3.75, 5.69, 4.11, 5.69, "starting weights", label_dy=0.09)
    arrow(8.63, 5.69, 8.98, 5.69, "adapted encoders", label_dy=0.09)
    arrow(8.10, 7.72, 8.10, 6.72, "SSL input", label_dx=0.42,
          label_dy=-0.04, color=TEAL, lw=0.95)

    # A baseline bypass makes the nine-model comparison topology explicit.
    ax.plot([2.13, 2.13, 10.72], [4.79, 4.67, 4.67], color=BLUE,
            lw=0.75, linestyle=(0, (3, 2)), zorder=3)
    arrow(10.72, 4.67, 10.72, 4.79, None, color=BLUE, lw=0.75)
    ax.text(6.35, 4.675, "unadapted baseline path", ha="center", va="bottom",
            fontsize=4.45, color=BLUE, zorder=4,
            bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.95))

    # ------------------------------------------------------------------
    # C. Evaluation framework: three complementary safeguards/evidence paths.
    # ------------------------------------------------------------------
    section("C", "Leakage-controlled evaluation framework", 1.63, 2.66,
            ORANGE, "#FBF6F1")
    card(0.50, 1.87, 3.78, 1.86, "Nested LODO benchmark", NAVY,
         title_fs=6.0)
    ax.text(2.39, 3.18, "1  Outer loop: hold out dataset d; train on remaining 3",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    ax.text(2.39, 2.80, "2  Inner train-side LODO selects checkpoint c*",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    ax.text(2.39, 2.42, "3  Lock c* → evaluate d exactly once",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    rounded(0.84, 1.98, 3.10, 0.26, fc="#EAF2F6", ec=BLUE,
            lw=0.55, radius=0.035, z=3)
    ax.text(2.39, 2.11, "AUROC / AUPRC  ·  patient-cluster bootstrap CI",
            ha="center", va="center", fontsize=4.45, color=NAVY, zorder=4)

    card(4.61, 1.87, 3.78, 1.86, "LOTO source-isolation audit", TEAL,
         title_fs=5.85)
    ax.text(6.50, 3.16, "For each target d: retrain on pretraining corpus − d",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    ax.text(6.50, 2.80, "Evaluate the resulting encoder on d",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    rounded(4.94, 2.31, 3.12, 0.30, fc="#E8F4F2", ec=TEAL,
            lw=0.55, radius=0.035, z=3)
    ax.text(6.50, 2.46, "transductive = full corpus − corpus-minus-d",
            ha="center", va="center", fontsize=4.35, color=TEAL,
            fontweight="bold", zorder=4)
    rounded(4.94, 1.98, 3.12, 0.26, fc="white", ec=TEAL,
            lw=0.55, radius=0.035, z=3)
    ax.text(6.50, 2.11, "inductive = corpus-minus-d − unadapted start",
            ha="center", va="center", fontsize=4.30, color=TEAL, zorder=4)

    card(8.72, 1.87, 3.75, 1.86, "Untouched external check", ORANGE,
         title_fs=6.0, fc="#FFFDFC")
    ax.text(10.595, 3.13, "HVDROPDB excluded from pretraining, probes,",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    ax.text(10.595, 2.80, "checkpoint selection, and threshold tuning",
            ha="center", va="center", fontsize=4.75, color=INK, zorder=4)
    ax.text(10.595, 2.39, "Evaluate all 15 frozen model variants once",
            ha="center", va="center", fontsize=4.8, fontweight="bold",
            color=ORANGE, zorder=4)
    rounded(9.05, 1.98, 3.09, 0.26, fc="#FFF4EC", ec=ORANGE,
            lw=0.55, radius=0.035, z=3)
    ax.text(10.595, 2.11, "image-level bootstrap CI  ·  no patient identifiers",
            ha="center", va="center", fontsize=4.25, color=ORANGE, zorder=4)

    arrow(6.50, 4.56, 6.50, 4.29, "evaluate", label_dx=0.42,
          label_dy=-0.05, color=INK, lw=0.95)

    # ------------------------------------------------------------------
    # Evidence portfolio: the breadth of the paper without obscuring the core.
    # ------------------------------------------------------------------
    rounded(0.08, 0.14, 12.84, 1.26, fc="#F6F7F8", ec=LINE,
            lw=0.75, radius=0.11, z=0)
    ax.text(0.40, 1.20, "Analysis portfolio", ha="left", va="center",
            fontsize=7.1, fontweight="bold", color=INK, zorder=3)
    portfolio = [
        (0.40, 2.78, "Primary discrimination", "AUROC / AUPRC  ·  decision units", NAVY),
        (3.31, 2.78, "Transfer & efficiency", "downstream probes  ·  label efficiency\nrepresentation structure", TEAL),
        (6.22, 2.78, "Robustness audits", "label audit  ·  shortcuts  ·  ablations\nseed sensitivity", RED),
        (9.13, 3.47, "Deployment simulation", "calibration  ·  locked thresholds\nrisk–coverage  ·  safe automation", ORANGE),
    ]
    for x, w, title, body, accent in portfolio:
        rounded(x, 0.34, w, 0.65, fc="white", ec=accent, lw=0.7,
                radius=0.055, z=2)
        ax.text(x + 0.13, 0.79, title, ha="left", va="center",
                fontsize=5.15, fontweight="bold", color=accent, zorder=4)
        ax.text(x + w / 2, 0.51, body, ha="center", va="center",
                fontsize=4.20, color=INK, linespacing=1.15, zorder=4)
    arrow(6.50, 1.63, 6.50, 1.40, "aggregate evidence", label_dx=0.56,
          label_dy=-0.05, color=INK, lw=0.9)

    fig.savefig(fig_dir / "Fig1_protocol.svg", facecolor="white",
                bbox_inches=None, pad_inches=0)
    fig.savefig(fig_dir / "Fig1_protocol.png", dpi=600, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print("Fig1 architecture ok (SVG + 600-dpi PNG)", flush=True)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "outputs/paper/figures")
