# Stage 6: look at the external glioma errors. Writes a grid of external gliomas missed by all six runs next to
# gliomas found by all six, and simple image statistics (mean, spread, foreground fraction) for both groups and for
# the internal test gliomas. Descriptive only: no cause is tested here.
# Usage: python scripts/12_error_examples.py RUNS_DIR DATA_NPZ EXTERNAL_NPZ OUT_DIR

import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_dir, data_path, ext_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
d, e = np.load(data_path, allow_pickle=True), np.load(ext_path, allow_pickle=True)
classes = ["glioma", "meningioma", "no tumour", "pituitary"]
rng = np.random.default_rng(0)

pred = np.stack([np.load(p)["prob"].argmax(1) for p in sorted(glob.glob(os.path.join(runs_dir, "*", "external_probs.npz")))])
gl = np.where(e["y"] == 0)[0]
wrong = gl[(pred[:, gl] != 0).all(0)]
right = gl[(pred[:, gl] == 0).all(0)]
mixed = len(gl) - len(wrong) - len(right)
print(f"external gliomas {len(gl)}: missed by all 6 runs {len(wrong)}, found by all 6 {len(right)}, mixed {mixed}")

img_int = d["images"][(d["split"] == "test") & (d["y"] == 0)]
groups = {"external, missed by all 6": e["images"][wrong], "external, found by all 6": e["images"][right], "internal test gliomas": img_int}


def stats(x):
    x = x.astype(np.float32)
    fg = (x > 20).mean((1, 2))  # fraction of pixels above a low threshold
    return x.mean((1, 2)), x.std((1, 2)), fg


with open(os.path.join(out, "glioma_image_stats.csv"), "w") as f:
    f.write("group,n,mean_intensity,sd_intensity,foreground_fraction\n")
    for k, x in groups.items():
        m, s, fg = stats(x)
        f.write(f"{k},{len(x)},{m.mean():.1f},{s.mean():.1f},{fg.mean():.3f}\n")
print(open(os.path.join(out, "glioma_image_stats.csv")).read())

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8})
n = 8
rows = [("(a) Missed by all six runs", wrong, True), ("(b) Found by all six runs", right, False)]
fig, axs = plt.subplots(2, n, figsize=(7.2, 2.7))
fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01, wspace=0.03, hspace=0.30)
for r, (title, ids, show) in enumerate(rows):
    pick = rng.choice(ids, n, replace=False)
    for c, i in enumerate(pick):
        ax = axs[r, c]
        ax.imshow(e["images"][i], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if show:
            ax.set_title(classes[int(np.bincount(pred[:, i], minlength=4).argmax())], fontsize=6)
    fig.text(0.01, axs[r, 0].get_position().y1 + (0.07 if show else 0.012), title + f" ({len(ids)} of {len(gl)} external gliomas)", fontweight="bold", fontsize=7.5, va="bottom")
for ext in ("png", "svg"):
    fig.savefig(os.path.join(out, "glioma_examples." + ext), dpi=300)
