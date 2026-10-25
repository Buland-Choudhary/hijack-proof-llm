"""Baseline charts from the core result files (static PNGs for the README).

    hp-charts            # writes results/baselines/asr_by_model.png and asr_by_family.png

Colors follow the data-viz palette: categorical slots 1-2 for the two setups (validated
colorblind-safe), the single-hue blue ramp for the heatmap, text in ink tokens (never in the
series color).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODELS = [  # (result-file stem prefix, display name)
    ("smollm2-135m", "SmolLM2-135M"),
    ("smollm2-360m", "SmolLM2-360M"),
    ("qwen3-0.6b", "Qwen3-0.6B"),
    ("qwen3-1.7b", "Qwen3-1.7B"),
    ("qwen3-4b-instruct", "Qwen3-4B*"),
]
SETUPS = [("none", "no defense", BLUE), ("sandwich", "'sandwich' reminder", ORANGE)]
HELDOUT = {"authority", "multilingual", "formatting", "obfuscation"}


def load(core_dir: Path = Path("results/core")) -> dict[tuple[str, str], dict]:
    out = {}
    for path in core_dir.glob("*__user_inline*.json"):
        d = json.loads(path.read_text())
        out[(d["meta"]["model"], d["meta"]["defense"])] = d["summary"]
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def asr_by_model(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    rows = [
        (name, color, results[(key, defense)]["asr_loose"])
        for key, name in MODELS
        for defense, _label, color in SETUPS
        if (key, defense) in results
    ]
    bar_h, gap, group_gap = 0.34, 0.12, 0.45
    y, last, centers = 0.0, None, {}
    for name, color, m in rows:
        if last is not None and name != last:
            y += group_gap
        last = name
        ax.barh(y, 100 * m["mean"], height=bar_h, color=color, zorder=2)
        lo, hi = 100 * (m["mean"] - m["lo"]), 100 * (m["hi"] - m["mean"])
        ax.errorbar(100 * m["mean"], y, xerr=[[lo], [hi]], color=INK2, capsize=2.5, lw=1, zorder=3)
        ax.text(
            100 * m["hi"] + 1.2, y, f"{100 * m['mean']:.0f}%", va="center", color=INK, fontsize=9
        )
        centers.setdefault(name, []).append(y)
        y += bar_h + gap
    ax.set_yticks([sum(v) / len(v) for v in centers.values()])
    ax.set_yticklabels(list(centers), color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, 81, 20))
    ax.set_xticklabels([f"{t}%" for t in range(0, 81, 20)], color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SETUPS]
    ax.legend(
        handles,
        [s[1] for s in SETUPS],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
    )
    none_rates = [
        100 * results[(k, "none")]["asr_loose"]["mean"] for k, _ in MODELS if (k, "none") in results
    ]
    fig.suptitle(
        "Untrained models obey hidden instructions in "
        f"{min(none_rates):.0f}-{max(none_rates):.0f}% of attacks",
        x=0.02, ha="left", color=INK, fontsize=13, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.02, 0.905,
        "Share of attacked prompts where the model did what the hidden text said "
        "(final test set, 95% interval)",
        color=INK2, fontsize=9,
    )  # fmt: skip
    fig.text(
        0.02, 0.01,
        "*Qwen3-4B was run on a third of the test items (450 of 1,350), no defense only.",
        color=INK2, fontsize=8,
    )  # fmt: skip
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def asr_by_family(results: dict, out: Path) -> None:
    cols = [(k, n) for k, n in MODELS if (k, "none") in results]
    fams = sorted(
        results[(cols[0][0], "none")]["attacked_slices"]["family"],
        key=lambda f: (f in HELDOUT, f == "multi", f),
    )
    grid = [
        [results[(k, "none")]["attacked_slices"]["family"][f]["asr_loose"] for k, _ in cols]
        for f in fams
    ]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.44 else INK)  # fmt: skip
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([n for _, n in cols], color=INK, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(
        [f"{f}  (held out)" if f in HELDOUT else f for f in fams], color=INK, fontsize=9
    )
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(fams))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    fig.suptitle(
        "Every attack style works; styles never seen in training work somewhat less",
        x=0.02, ha="left", color=INK, fontsize=13, fontweight="bold",
    )  # fmt: skip
    fig.text(0.02, 0.905, "Attack success rate (%) by attack style, no defense, final test set. "
             "Darker = more often obeyed.", color=INK2, fontsize=9)  # fmt: skip
    fig.text(
        0.02, 0.01,
        "'held out' styles were never used to build training data. *Qwen3-4B: 450 of 1,350 items.",
        color=INK2, fontsize=8,
    )  # fmt: skip
    fig.subplots_adjust(left=0.27, right=0.98, top=0.8, bottom=0.07)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    results = load()
    out_dir = Path("results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    asr_by_model(results, out_dir / "asr_by_model.png")
    asr_by_family(results, out_dir / "asr_by_family.png")
    print(f"wrote charts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
