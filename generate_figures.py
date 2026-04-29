"""
generate_figures.py
===================
Generate all six report figures (Fig. 1 – Fig. 6) for the CAES9541 final report.

Figures produced
----------------
Fig. 1  System architecture schematic                  → figure1_system_architecture.{png,pdf}
Fig. 2  Max-zone 5×5 zone-to-class assignment map      → figure3_zone_assignment.{png,pdf}
Fig. 3  Training and validation accuracy curves         → figure4_training_curves.{png,pdf}
Fig. 4  Per-class test accuracy bar chart               → figure5_per_class_accuracy.{png,pdf}
Fig. 5  Row-normalised confusion matrix                 → figure6_confusion_matrix.{png,pdf}
Fig. 6  Trained amplitude masks (Layer 1 & Layer 2)    → figure7_masks.{png,pdf}

Output filenames follow the report's figure embed paths (report/figures/).

Run from the ONN project root:
    python generate_figures.py              # both sets
    python generate_figures.py --report     # report figures only  (with IEEE captions)
    python generate_figures.py --presentation  # presentation figures only (no captions)
"""

import sys
import csv
import argparse
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── project root on path so config/model/etc. are importable ─────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from model  import OpticalNeuralNetwork

# ── paths ─────────────────────────────────────────────────────────────────────
CKPT     = ROOT / "checkpoints_maxzone_5x5" / "best.pt"
LOG_CSV  = ROOT / "checkpoints_maxzone_5x5" / "training_log.csv"
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "report" / "figures"
PRES_DIR = ROOT / "report" / "figures_presentation"
POSTER_DIR = ROOT / "report" / "figures_poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRES_DIR.mkdir(parents=True, exist_ok=True)
POSTER_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── IEEE-compliant matplotlib style ──────────────────────────────────────────
# Single-column IEEE figure: 3.5 in wide; double-column: 7.16 in wide.
# Font: Times New Roman or equivalent serif; minimum 8 pt in caption.
plt.rcParams.update({
    "font.family":      "DejaVu Serif",     # closest freely available to Times
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.linewidth":   0.7,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.45,
    "lines.linewidth":  1.3,
})

# ── colour palette (IEEE-friendly, print-safe) ────────────────────────────────
C_BLUE   = "#2166AC"
C_ORANGE = "#D6604D"
C_GREEN  = "#4DAC26"
C_GREY   = "#777777"


# ── global mode flag (set by main) ────────────────────────────────────────────
_PRESENTATION_MODE = False
_POSTER_MODE       = False
_FIGSIZE_SCALE     = 1.0      # set to 2.5 for --poster
_SAVE_OVERRIDES = {}   # {original_stem: (new_stem, new_caption)}


def _fs(size: float) -> float:
    """Scale a hardcoded fontsize by _FIGSIZE_SCALE when in poster mode."""
    return size * _FIGSIZE_SCALE if _POSTER_MODE else size


def _save(fig, stem: str, caption: str = ""):
    """Save figure as both PNG and PDF; add IEEE caption below if provided."""
    if _POSTER_MODE:
        out = POSTER_DIR
    elif _PRESENTATION_MODE:
        out = PRES_DIR
    else:
        out = OUT_DIR
    if stem in _SAVE_OVERRIDES:
        stem, caption = _SAVE_OVERRIDES[stem]
    if caption:
        wrap_w = int(fig.get_size_inches()[0] * 12)
        wrapped = textwrap.fill(caption, width=max(wrap_w, 60))
        fig.text(0.5, -0.04, wrapped, ha="center", va="top",
                 fontsize=_fs(8), fontstyle="normal")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 1 — System Architecture Schematic
# ══════════════════════════════════════════════════════════════════════════════
def plot_system_architecture():
    """
    IEEE double-column width (7.16 in).  Programmatically drawn — no external
    image dependency.  All boxes use identical dimensions.
    """
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    C_SRC = "#2166AC"
    C_LCD = "#4DAC26"
    C_DET = "#D6604D"

    BOX_W = 1.5
    BOX_H = 1.3

    def box(cx, cy, colour, label, sublabel=""):
        rect = FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.06",
            linewidth=0.8, edgecolor="#222222", facecolor=colour, zorder=3,
        )
        ax.add_patch(rect)
        dy = 0.12 if sublabel else 0
        ax.text(cx, cy + dy, label, ha="center", va="center",
                fontsize=_fs(8), fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.25, sublabel, ha="center", va="center",
                    fontsize=_fs(6.5), color="white", zorder=4, alpha=0.9)

    def arrow(x0, x1, y=2.0):
        ax.annotate(
            "", xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.9), zorder=2,
        )

    # component centres, evenly spaced
    xs = [1.2, 3.4, 5.6, 7.8, 10.0]
    labels_   = ["532 nm\nLaser", "Input\nPlane", "LCD\nLayer 1",
                 "LCD\nLayer 2", "Detector\nPlane"]
    subs      = ["", "(MNIST digit)", "(200\u00d7200 px)",
                 "(200\u00d7200 px)", "(max-zone 5\u00d75)"]
    colours   = [C_SRC, C_LCD, C_LCD, C_LCD, C_DET]

    for cx, col, lbl, sub in zip(xs, colours, labels_, subs):
        box(cx, 2.0, col, lbl, sub)

    # arrows between boxes
    gap = BOX_W / 2 + 0.08
    for i in range(len(xs) - 1):
        arrow(xs[i] + gap, xs[i + 1] - gap)

    # propagation distance labels
    for mid_x in [(xs[1] + xs[2]) / 2, (xs[2] + xs[3]) / 2]:
        ax.text(mid_x, 3.0, "50 mm", ha="center", va="bottom",
                fontsize=_fs(7), color="#444444", style="italic")
        ax.plot([mid_x - 0.4, mid_x + 0.4], [2.92, 2.92],
                color="#aaaaaa", lw=0.6)

    # beam dashed line
    ax.plot([0.3, 11.0], [2.0, 2.0], color="#aab8cc", lw=0.7,
            linestyle="--", zorder=1, alpha=0.45)

    # legend
    patches = [
        mpatches.Patch(color=C_SRC, label="Coherent source"),
        mpatches.Patch(color=C_LCD, label="Amplitude-modulating LCD"),
        mpatches.Patch(color=C_DET, label="Intensity detector (CMOS)"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=3,
              fontsize=_fs(7.5), framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    _save(fig, "figure1_system_architecture",
          "Fig. 1.  System architecture of the two-layer LCD-based ONN.")
    print("  [OK] Fig. 1 — system architecture")


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 2 — Zone Assignment Map  (model-driven, no file copy)
# ══════════════════════════════════════════════════════════════════════════════
def plot_zone_assignment(model, rows=5, cols=5):
    """
    Renders the learned zone-to-class assignment map
    directly from the trained model weights.  Single panel, black text.
    """
    zone_assign = model.detector.get_zone_assignment()   # {z: (digit, weight)}

    num_zones = rows * cols
    digit_map = np.zeros((rows, cols), dtype=int)
    for z in range(num_zones):
        r, c = divmod(z, cols)
        digit_map[r, c] = zone_assign[z][0]

    # 10 easily-distinguished colours, one per digit class
    CLASS_COLOURS = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#A65628", "#F781BF", "#999999", "#FFFF33", "#00CED1",
    ]
    cmap_disc = matplotlib.colors.ListedColormap(CLASS_COLOURS)

    fig, ax = plt.subplots(figsize=(4.0, 3.8))

    im = ax.imshow(digit_map, cmap=cmap_disc, vmin=-0.5, vmax=9.5,
                   interpolation="nearest")
    for z in range(num_zones):
        r, c = divmod(z, cols)
        ax.text(c, r, f"Zone {z}\nClass {zone_assign[z][0]}",
                ha="center", va="center", fontsize=_fs(7.5),
                color="black", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, ticks=range(10), fraction=0.046, pad=0.04)
    cb.set_label("Digit class", fontsize=_fs(8))
    cb.ax.tick_params(labelsize=_fs(7))

    fig.tight_layout()
    _save(fig, "figure3_zone_assignment",
          "Fig. 2.  Max-zone 5 \u00d7 5 detector zone-to-class"
          " assignment map after convergence at epoch 373.")
    print("  [OK] Fig. 2 — zone assignment map")


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 3 — Training Curves  (from real CSV log)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    epochs, train_acc, val_acc, mask_lr = [], [], [], []
    lr_reductions = []

    with open(LOG_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    prev_lr = None
    for row in rows:
        ep  = int(row["epoch"])
        mlr = float(row["mask_lr"])
        epochs.append(ep)
        train_acc.append(float(row["train_acc"]))
        val_acc.append(float(row["val_acc"]))
        mask_lr.append(mlr)
        if prev_lr is not None and mlr < prev_lr * 0.6:
            lr_reductions.append(ep)
        prev_lr = mlr

    epochs    = np.array(epochs)
    train_acc = np.array(train_acc)
    val_acc   = np.array(val_acc)
    mask_lr   = np.array(mask_lr)

    best_val = val_acc.max()
    best_ep  = epochs[val_acc.argmax()]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.16, 4.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # accuracy panel
    ax1.plot(epochs, train_acc, color=C_BLUE,   lw=1.3, label="Training accuracy")
    ax1.plot(epochs, val_acc,   color=C_ORANGE, lw=1.3, label="Validation accuracy")

    for ep in lr_reductions:
        ax1.axvline(ep, color=C_GREY, lw=0.6, linestyle="--", alpha=0.6)

    ax1.axhline(best_val, color=C_ORANGE, lw=0.7, linestyle=":", alpha=0.85)
    ax1.annotate(
        f"Best val: {best_val:.2f}%  (epoch {best_ep - 1})",
        xy=(best_ep, best_val),
        xytext=(best_ep - epochs[-1] * 0.35, best_val - 9),
        arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.8),
        fontsize=8, color=C_ORANGE,
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax1.legend(loc="lower right")
    ax1.grid(axis="y")

    lr_handle = Line2D([0], [0], color=C_GREY, lw=0.8,
                       linestyle="--", label="Plateau LR reduction")
    ax1.legend(
        handles=[
            Line2D([0], [0], color=C_BLUE,   lw=1.3, label="Training accuracy"),
            Line2D([0], [0], color=C_ORANGE, lw=1.3, label="Validation accuracy"),
            lr_handle,
        ],
        loc="lower right", fontsize=8,
    )

    # LR panel
    ax2.semilogy(epochs, mask_lr, color=C_GREEN, lw=1.2)
    for ep in lr_reductions:
        ax2.axvline(ep, color=C_GREY, lw=0.6, linestyle="--", alpha=0.6)
    ax2.set_ylabel("LR (mask)")
    ax2.set_xlabel("Epoch")
    ax2.grid(axis="y")
    ax2.legend(handles=[lr_handle], fontsize=8, loc="upper right")

    fig.tight_layout()
    _save(fig, "figure4_training_curves",
          f"Fig. 3.  Training and validation accuracy curves for the"
          f" max-zone 5 \u00d7 5 model over {epochs[-1] - 1} epochs, with"
          f" the learning rate schedule shown in the lower panel.")
    print(f"  [OK] Fig. 3 — training curves  (best val {best_val:.2f}% @ epoch {best_ep - 1})")


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 4 — Per-Class Test Accuracy  (model-driven)
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_accuracy(model):
    cfg         = Config()
    test_ds     = datasets.MNIST(DATA_DIR, train=False, download=False,
                                 transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    correct = np.zeros(10, dtype=int)
    total   = np.zeros(10, dtype=int)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs     = imgs.to(DEVICE)
            labels   = labels.to(DEVICE)
            imgs_up  = torch.nn.functional.interpolate(
                imgs, size=(140, 140), mode="bilinear", align_corners=False)
            imgs_pad = torch.nn.functional.pad(imgs_up, (30, 30, 30, 30))
            logits   = model(imgs_pad.squeeze(1))
            preds    = logits.argmax(dim=1)
            for d in range(10):
                mask        = labels == d
                total[d]   += mask.sum().item()
                correct[d] += (preds[mask] == labels[mask]).sum().item()

    per_class = correct / total * 100
    overall   = correct.sum() / total.sum() * 100
    print(f"     Overall: {overall:.2f}%")
    for d in range(10):
        print(f"     Digit {d}: {per_class[d]:.1f}%  ({correct[d]}/{total[d]})")

    labels_bar = [str(d) for d in range(10)]

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    bars = ax.bar(labels_bar, per_class, color=C_BLUE,
                  edgecolor="white", linewidth=0.4, width=0.65)
    for bar, v in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")

    ax.set_xlabel("Digit class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.grid(axis="y")

    fig.tight_layout()
    _save(fig, "figure5_per_class_accuracy",
          "Fig. 4.  Per-class test accuracy of the max-zone 5 \u00d7 5"
          " model on the 10,000-image MNIST test set.")
    print("  [OK] Fig. 4 — per-class accuracy")
    return per_class, overall


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 5 — Confusion Matrix  (model-driven)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(model):
    test_ds     = datasets.MNIST(DATA_DIR, train=False, download=False,
                                 transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs     = imgs.to(DEVICE)
            imgs_up  = torch.nn.functional.interpolate(
                imgs, size=(140, 140), mode="bilinear", align_corners=False)
            imgs_pad = torch.nn.functional.pad(imgs_up, (30, 30, 30, 30))
            logits   = model(imgs_pad.squeeze(1))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised accuracy (%)", fontsize=_fs(8))
    cb.ax.tick_params(labelsize=_fs(7))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    for i in range(10):
        for j in range(10):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=_fs(6.5), color="white" if v > 55 else "black")

    fig.tight_layout()
    _save(fig, "figure6_confusion_matrix",
          "Fig. 5.  Row-normalised confusion matrix for the max-zone"
          " 5 \u00d7 5 model on the 10,000-image MNIST test set.")
    print("  [OK] Fig. 5 — confusion matrix")


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 6 — Trained Amplitude Masks  (model-driven)
# ══════════════════════════════════════════════════════════════════════════════
def plot_masks(model):
    with torch.no_grad():
        masks = [torch.sigmoid(layer.raw.detach().cpu()).numpy()
                 for layer in model.layers]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))
    fig.subplots_adjust(wspace=0.18)

    for ax, mask, title in zip(axes, masks, ["(a) Layer 1", "(b) Layer 2"]):
        img = (mask * 255).astype(np.uint8)
        im  = ax.imshow(img, cmap="gray", vmin=0, vmax=255,
                        interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Grey level (0 = blocking, 255 = transmitting)", fontsize=7)
        cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "figure7_masks",
          "Fig. 6.  Trained amplitude masks for Layer 1 (left) and"
          " Layer 2 (right) of the max-zone 5 \u00d7 5 model after"
          " convergence at epoch 373.")
    print("  [OK] Fig. 6 — amplitude masks")


# ══════════════════════════════════════════════════════════════════════════════
# Presentation — Propagation through the ONN for a single digit
# ══════════════════════════════════════════════════════════════════════════════
def plot_propagation(model, digit=6):
    """
    Single-row figure showing light intensity at each stage:
    Input → Mask 1 → After Mask 1 → After Propagation → Mask 2 → After Mask 2 → Detector intensity
    Uses the model's forward pass with return_intermediate=True.
    """
    # Get one sample of the requested digit
    test_ds = datasets.MNIST(DATA_DIR, train=True, download=False,
                             transform=transforms.ToTensor())
    for img_tensor, label in test_ds:
        if label == digit:
            break

    # Preprocess: upsample and pad to 200×200 (same as training pipeline)
    img = img_tensor.unsqueeze(0)                                         # [1,1,28,28]
    img = torch.nn.functional.interpolate(img, size=(140, 140),
                                          mode="bilinear", align_corners=False)
    img = torch.nn.functional.pad(img, (30, 30, 30, 30))                  # [1,1,200,200]
    img = img.to(DEVICE)

    # Forward pass capturing intermediates
    model.eval()
    with torch.no_grad():
        logits, intermediates = model(img, return_intermediate=True)
    pred = logits.argmax(dim=1).item()

    # intermediates: [input_field, after_mask1, after_prop1, after_mask2, after_prop2]
    # after_prop2 is the camera field
    input_field     = intermediates[0][0].cpu()
    after_mask1     = intermediates[1][0].cpu()
    after_prop1     = intermediates[2][0].cpu()
    after_mask2     = intermediates[3][0].cpu()
    camera_field    = intermediates[4][0].cpu()

    # Get masks
    masks = model.get_masks()
    mask1 = masks[0].cpu().numpy()
    mask2 = masks[1].cpu().numpy()

    # Compute intensities
    input_int  = torch.abs(input_field) ** 2
    am1_int    = torch.abs(after_mask1) ** 2
    ap1_int    = torch.abs(after_prop1) ** 2
    am2_int    = torch.abs(after_mask2) ** 2
    cam_int    = torch.abs(camera_field) ** 2

    # Zone intensity from camera
    zone_grid = _zone_intensity_grid(model, cam_int)

    # ── Build the figure: 8 panels in a row ──
    titles = [
        "Input\n(digit image)",
        "Mask 1",
        "After\nMask 1",
        "After\nPropagation",
        "Mask 2",
        "After\nMask 2",
        "Detector",
        "Zone\nIntensity",
    ]
    fig, axes = plt.subplots(1, 8, figsize=(7.16 * 2, 2.6))
    fig.subplots_adjust(wspace=0.12)

    def _show(ax, data, cmap="hot", title="", is_mask=False):
        if is_mask:
            ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(data.numpy(), cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=_fs(8), pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared intensity vmax (exclude zone grid)
    vmax = max(input_int.max(), am1_int.max(), ap1_int.max(),
               am2_int.max(), cam_int.max()).item()

    def _show_int(ax, data, title):
        ax.imshow(data.numpy(), cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=_fs(8), pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    _show_int(axes[0], input_int, titles[0])
    _show(axes[1], mask1, title=titles[1], is_mask=True)
    _show_int(axes[2], am1_int, titles[2])
    _show_int(axes[3], ap1_int, titles[3])
    _show(axes[4], mask2, title=titles[4], is_mask=True)
    _show_int(axes[5], am2_int, titles[5])
    _show_int(axes[6], cam_int, titles[6])

    # Zone intensity grid (separate colour scale)
    im = axes[7].imshow(zone_grid, cmap="hot", interpolation="nearest")
    axes[7].set_title(titles[7], fontsize=8, pad=4)
    axes[7].set_xticks([])
    axes[7].set_yticks([])
    # Annotate zone values
    rows_z, cols_z = zone_grid.shape
    zmax = zone_grid.max()
    for r in range(rows_z):
        for c in range(cols_z):
            v = zone_grid[r, c]
            colour = "white" if v < zmax * 0.6 else "black"
            axes[7].text(c, r, f"{v:.1e}", ha="center", va="center",
                         fontsize=5, color=colour)
    cb = fig.colorbar(im, ax=axes[7], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)

    fig.tight_layout()
    _save(fig, f"propagation_digit{digit}",
          f"Light propagation through the two-layer ONN for digit {digit}"
          f" (predicted: {pred}).")
    print(f"  [OK] Propagation — digit {digit} (pred {pred})")


def _zone_intensity_grid(model, camera_intensity, rows=5, cols=5):
    """Compute average intensity per zone from detector field, return as grid."""
    h, w = camera_intensity.shape
    zh, zw = h // rows, w // cols
    grid = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            zone = camera_intensity[r*zh:(r+1)*zh, c*zw:(c+1)*zw]
            grid[r, c] = zone.mean().item()
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    cfg   = Config()
    model = OpticalNeuralNetwork(
        cfg, detection_method="maxzone", rows=5, cols=5
    ).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"  Model loaded — epoch {ck['epoch']}, best_acc {ck['best_acc']:.2f}%")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate report / presentation figures")
    parser.add_argument("--report",       action="store_true", help="Report figures only")
    parser.add_argument("--presentation", action="store_true", help="Presentation figures only")
    parser.add_argument("--poster",       action="store_true", help="High-res poster figures (2.5x figsize, 300 DPI)")
    args = parser.parse_args()

    do_report = args.report or (not args.report and not args.presentation and not args.poster)
    do_pres   = args.presentation or (not args.report and not args.presentation and not args.poster)
    do_poster = args.poster

    _model = [None]
    def _ensure_model():
        if _model[0] is None:
            print("\n[Loading model] …")
            _model[0] = load_model()
        return _model[0]

    # ── Report figures (IEEE captions → report/figures/) ──
    if do_report:
        _PRESENTATION_MODE = False
        print("=" * 60)
        print("Generating REPORT figures (IEEE style)")
        print("=" * 60)

        print("\n[Fig. 1] System architecture …")
        plot_system_architecture()

        m = _ensure_model()

        print("\n[Fig. 2] Zone assignment map …")
        plot_zone_assignment(m)

        print("\n[Fig. 3] Training curves …")
        plot_training_curves()

        print("\n[Fig. 4] Per-class accuracy …")
        plot_per_class_accuracy(m)

        print("\n[Fig. 5] Confusion matrix …")
        plot_confusion_matrix(m)

        print("\n[Fig. 6] Amplitude masks …")
        plot_masks(m)

        print(f"\nReport figures → {OUT_DIR}")

    # ── Presentation figures (with captions → report/figures_presentation/) ──
    if do_pres:
        _PRESENTATION_MODE = True
        _SAVE_OVERRIDES.clear()
        _SAVE_OVERRIDES.update({
            "figure1_system_architecture": (
                "pres_fig1_system_architecture",
                "Fig. 1.  System architecture of the two-layer LCD-based ONN."),
            "propagation_digit6": (
                "pres_fig2_propagation",
                "Fig. 2.  Light propagation through the two-layer ONN for digit 6."),
            "figure3_zone_assignment": (
                "pres_fig3_zone_assignment",
                "Fig. 3.  Max-zone 5 \u00d7 5 detector zone-to-class assignment map"
                " after convergence at epoch 373."),
            "figure7_masks": (
                "pres_fig4_masks",
                "Fig. 4.  Trained amplitude masks for Layer 1 (left) and"
                " Layer 2 (right) of the max-zone 5 \u00d7 5 model."),
            "figure6_confusion_matrix": (
                "pres_fig5_confusion_matrix",
                "Fig. 5.  Row-normalised confusion matrix for the max-zone"
                " 5 \u00d7 5 model on the 10,000-image MNIST test set."),
        })
        print("\n" + "=" * 60)
        print("Generating PRESENTATION figures")
        print("=" * 60)

        print("\n[Fig. 1] System architecture …")
        plot_system_architecture()

        m = _ensure_model()

        print("\n[Fig. 2] Propagation …")
        plot_propagation(m, digit=6)

        print("\n[Fig. 3] Zone assignment …")
        plot_zone_assignment(m)

        print("\n[Fig. 4] Amplitude masks …")
        plot_masks(m)

        print("\n[Fig. 5] Confusion matrix …")
        plot_confusion_matrix(m)

        _SAVE_OVERRIDES.clear()
        print(f"\nPresentation figures → {PRES_DIR}")

    # ── Poster figures (high-res 2.5× figsize → report/figures_poster/) ──
    if do_poster:
        _PRESENTATION_MODE = False
        _POSTER_MODE       = True
        _FIGSIZE_SCALE     = 2.5

        # Scale rcParams for figures that use standard axis labels (e.g. confusion matrix)
        plt.rcParams.update({
            "savefig.dpi":      300,
            "font.size":        9  * _FIGSIZE_SCALE,
            "axes.titlesize":   9  * _FIGSIZE_SCALE,
            "axes.labelsize":   9  * _FIGSIZE_SCALE,
            "xtick.labelsize":  8  * _FIGSIZE_SCALE,
            "ytick.labelsize":  8  * _FIGSIZE_SCALE,
            "legend.fontsize":  8  * _FIGSIZE_SCALE,
            "axes.linewidth":   0.7 * _FIGSIZE_SCALE,
            "grid.linewidth":   0.4 * _FIGSIZE_SCALE,
            "lines.linewidth":  1.3 * _FIGSIZE_SCALE,
        })
        # _fs() helper uses _FIGSIZE_SCALE for hardcoded fontsize= calls in each function

        # Monkey-patch plt.subplots to scale all figsize arguments
        _orig_subplots = plt.subplots
        def _scaled_subplots(*args, **kwargs):
            if "figsize" in kwargs:
                w, h = kwargs["figsize"]
                kwargs["figsize"] = (w * _FIGSIZE_SCALE, h * _FIGSIZE_SCALE)
            return _orig_subplots(*args, **kwargs)
        plt.subplots = _scaled_subplots

        _SAVE_OVERRIDES.clear()
        _SAVE_OVERRIDES.update({
            "figure1_system_architecture": (
                "poster_fig1_system_architecture",
                "Fig. 1.  System architecture of the two-layer LCD-based ONN."),
            "propagation_digit6": (
                "poster_fig2_propagation",
                "Fig. 2.  Light propagation through the two-layer ONN for digit 6."),
            "figure3_zone_assignment": (
                "poster_fig3_zone_assignment",
                "Fig. 3.  Max-zone 5 \u00d7 5 detector zone-to-class assignment map."),
            "figure6_confusion_matrix": (
                "poster_fig5_confusion_matrix",
                "Fig. 5.  Row-normalised confusion matrix for the max-zone 5 \u00d7 5 model on the 10,000-image MNIST test set."),
        })
        print("\n" + "=" * 60)
        print("Generating POSTER figures (2.5× size, 300 DPI)")
        print("=" * 60)

        print("\n[Fig. 1] System architecture …")
        plot_system_architecture()

        m = _ensure_model()

        print("\n[Fig. 2] Propagation …")
        plot_propagation(m, digit=6)

        print("\n[Fig. 3] Zone assignment …")
        plot_zone_assignment(m)

        print("\n[Fig. 5] Confusion matrix …")
        plot_confusion_matrix(m)

        plt.subplots = _orig_subplots   # restore
        _SAVE_OVERRIDES.clear()
        print(f"\nPoster figures → {POSTER_DIR}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
