"""WP3 report figure: the four half-split kernels (TB, LR, DR, UR) for 4x4 and 3x3 windows.

Weights come from kernel_features.make_kernels (row 0 = south, drawn north up).
Red = half A (+), blue = half B (-), white = weight 0; each cell shows its weight.

Run: python scripts/plot_kernel_shapes.py [output.png]
Default output: outputs/figures/kernel_shapes.png
"""
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_features import make_kernels, KERNEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITLES = {"TB": "TB: north − south", "LR": "LR: east − west",
          "DR": "DR: NE − SW (diagonal ↘)", "UR": "UR: NW − SE (diagonal ↗)"}
CMAP = ListedColormap(["#4a7fc1", "#ffffff", "#d6604d"])  # half B (−), 0, half A (+)


def main(out):
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.0))
    for r, n in enumerate([4, 3]):
        kern = make_kernels(n)
        for c, name in enumerate(KERNEL_NAMES):
            ax = axes[r, c]
            w = kern[name]
            ax.imshow((w > 0).astype(int) - (w < 0).astype(int), origin="lower", cmap=CMAP, vmin=-1, vmax=1)
            for i in range(n):
                for j in range(n):
                    v = w[i, j]
                    txt = "0" if v == 0 else ("+" if v > 0 else "−") + str(Fraction(abs(v)).limit_denominator(16))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=12 if n == 4 else 14)
            for k in range(n + 1):
                ax.axhline(k - 0.5, color="black", linewidth=1, zorder=3)
                ax.axvline(k - 0.5, color="black", linewidth=1, zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(TITLES[name], fontsize=11)
        axes[r, 0].set_ylabel(f"{n}×{n} window", fontsize=12)
    fig.text(0.5, 0.02, "Red = half A (+), blue = half B (−), white = weight 0.  "
             "Kernel value = mean(half A) − mean(half B).  North up, east right.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "outputs" / "figures" / "kernel_shapes.png")
