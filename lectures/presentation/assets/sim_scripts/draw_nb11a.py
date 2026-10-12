import os, sys, pickle as pkl
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
ROOT = r"C:\Users\Eshel\workspace\sim_course"
OUT = ROOT + r"\lectures\presentation\assets\nb11"
NAVY, BLUE, GREEN, ORANGE, RED, GREY = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#C0392B", "#5D6D7E"
HIST = "#AFC8DD"
plt.rcParams.update({"font.size": 14})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)
def ph_pdf(x, a, S):
    s = -S.sum(axis=1)
    return np.array([a @ expm(S * xi) @ s for xi in x])

# 1. call center
d = pd.read_csv(ROOT + r"\data\call_center.csv")["duration"].values
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(d[d <= 2200], bins=440, density=True, color=BLUE)
ax.set_xlim(0, 2200); ax.set_ylim(0, 0.0065); ax.set_xlabel("call duration (seconds)"); ax.set_ylabel("density"); clean(ax)
ax.text(0.97, 0.9, f"{len(d):,} calls\nmean {d.mean():.0f} s\nSCV {d.var()/d.mean()**2:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=14, color=NAVY)
save(fig, "callcenter.png")

# 2. true density + histogram, 3. exponential fit
a, S, s = pkl.load(open(ROOT + r"\pkls\fitting_example.pkl", "rb"))
a = np.asarray(a).ravel(); S = np.asarray(S)
x = np.linspace(0.001, 4, 160); y = ph_pdf(x, a, S)
np.save(os.path.join(OUT, "_true_pdf.npy"), np.vstack([x, y]))
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(s, bins=300, density=True, color=HIST, label="50,000 samples")
ax.plot(x, y, color=NAVY, lw=3, label="true density")
ax.set_xlim(0, 4); ax.set_xlabel("service time"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
save(fig, "true_hist.png")

m = s.mean()
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(s, bins=300, density=True, color=HIST, label="data")
ax.plot(x, y, color=NAVY, lw=2.5, ls="--", label="true density")
ax.plot(x, np.exp(-x / m) / m, color=RED, lw=3, label=f"exponential fit, rate 1/{m:.3f}")
ax.set_xlim(0, 4); ax.set_xlabel("service time"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
save(fig, "exp_fit.png")
print("ok", m, s.var() / m**2)
