import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from scipy.stats import norm
OUT = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets\nb12"
NAVY, BLUE, GREEN, ORANGE, RED, GREY = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#C0392B", "#5D6D7E"
HIST = "#AFC8DD"
plt.rcParams.update({"font.size": 14})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)

# 1. LCG cycle
seq = [7]
for _ in range(16): seq.append((5 * seq[-1] + 3) % 16)
cyc = seq[:16]
fig, ax = plt.subplots(figsize=(6.4, 6.4)); ax.set_aspect("equal"); ax.axis("off")
R = 2.6; pos = {}
for i, v in enumerate(cyc):
    th = np.pi / 2 - 2 * np.pi * i / 16
    pos[v] = (R * np.cos(th), R * np.sin(th))
for i, v in enumerate(cyc):
    w = cyc[(i + 1) % 16]
    ax.add_patch(FancyArrowPatch(pos[v], pos[w], arrowstyle="-|>", mutation_scale=14, lw=1.6, color=GREY,
                                 shrinkA=17, shrinkB=17, connectionstyle="arc3,rad=-0.12"))
for v, (x, y) in pos.items():
    c = ORANGE if v == 7 else (GREEN if v == 8 else NAVY)
    ax.add_patch(Circle((x, y), 0.36, fc="white", ec=c, lw=2.6 if v in (7, 8) else 1.8, zorder=3))
    ax.text(x, y, str(v), ha="center", va="center", fontsize=15, fontweight="bold", color=c, zorder=4)
for v, lab, c in [(7, "seed 7", ORANGE), (8, "seed 8", GREEN)]:
    x, y = pos[v]; ax.text(x * 1.33, y * 1.33, lab, ha="center", va="center", fontsize=14, color=c, fontweight="bold")
ax.text(0, 0, "16 values,\nthen it repeats", ha="center", va="center", fontsize=16, color=NAVY)
ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.6, 3.6)
save(fig, "lcg_cycle.png")
print("cycle from 7:", cyc)

# 2. uniform histogram
rng = np.random.default_rng(42); u = rng.random(1_000_000)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.hist(u[:100_000], bins=30, density=True, color=HIST, edgecolor="white")
ax.axhline(1, color=NAVY, lw=2.5, ls="--", label="density = 1")
ax.set_xlabel("u"); ax.set_ylabel("density"); ax.set_ylim(0, 1.25); ax.legend(frameon=False, loc="upper right"); clean(ax)
save(fig, "uniform_hist.png")

# 3. partition of [0,1]
fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
segs = [(0, 0.2, "X = 10", BLUE), (0.2, 0.7, "X = 20", GREEN), (0.7, 1.0, "X = 50", ORANGE)]
for a, b, lab, c in segs:
    ax.add_patch(Rectangle((a, 0), b - a, 0.5, fc=c, alpha=0.25, ec=c, lw=2))
    ax.text((a + b) / 2, 0.25, lab, ha="center", va="center", fontsize=17, color=c, fontweight="bold")
    ax.text((a + b) / 2, -0.28, f"length {b - a:.1f}", ha="center", fontsize=13, color=GREY)
for t in [0, 0.2, 0.7, 1.0]:
    ax.text(t, 0.62, f"{t:g}", ha="center", fontsize=14, color=NAVY)
for uu in [0.13, 0.46, 0.83]:
    ax.annotate("", (uu, 0.5), (uu, 1.05), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(uu, 1.12, f"U = {uu}", ha="center", fontsize=13, color=RED)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.4, 1.3)
save(fig, "partition.png")

# 4. inverse transform on exponential CDF
lam = 2.0
x = np.linspace(0, 2.5, 400)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x, 1 - np.exp(-lam * x), color=NAVY, lw=3, label=r"$F(x) = 1 - e^{-2x}$")
for uu, c in [(0.25, GREEN), (0.6, ORANGE), (0.9, RED)]:
    xx = -np.log(1 - uu) / lam
    ax.annotate("", (xx, uu), (0, uu), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.annotate("", (xx, 0), (xx, uu), arrowprops=dict(arrowstyle="-|>", color=c, lw=2, ls="--"))
    ax.text(-0.05, uu, f"U = {uu}", ha="right", va="center", fontsize=13, color=c)
    ax.text(xx, -0.07, f"{xx:.2f}", ha="center", va="top", fontsize=13, color=c)
ax.set_xlim(-0.45, 2.5); ax.set_ylim(-0.15, 1.05)
ax.set_xlabel("x", labelpad=18); ax.set_yticks([0, 0.5, 1]); ax.legend(frameon=False, loc="lower right"); clean(ax)
save(fig, "inverse_transform.png")

# 5. manual vs numpy
rng1 = np.random.default_rng(100); rng2 = np.random.default_rng(200)
n = 200_000
xm = -np.log(rng1.random(n)) / lam; xn = rng2.exponential(1 / lam, n)
fig, ax = plt.subplots(figsize=(10, 4.4))
bins = np.linspace(0, 4, 81)
ax.hist(xm, bins=bins, density=True, histtype="step", lw=2.2, color=BLUE, label=f"inverse transform (mean {xm.mean():.4f})")
ax.hist(xn, bins=bins, density=True, histtype="step", lw=2.2, color=ORANGE, label=f"NumPy (mean {xn.mean():.4f})")
xx = np.linspace(0, 4, 300); ax.plot(xx, lam * np.exp(-lam * xx), color=NAVY, lw=1.6, ls="--", label="true density")
ax.set_xlim(0, 3); ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
save(fig, "manual_vs_numpy.png")

# 6. Box-Muller
rng = np.random.default_rng(1234); n = 500_000
u1, u2 = rng.random(n), rng.random(n)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.hist(z1, bins=80, density=True, color=HIST, edgecolor="white", label="Box–Muller samples")
zz = np.linspace(-4.5, 4.5, 300); ax.plot(zz, norm.pdf(zz), color=NAVY, lw=2.5, label="N(0,1) density")
ax.set_xlim(-4.5, 4.5); ax.set_xlabel("z"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
save(fig, "box_muller.png")

# 7. M/M/1 inputs from uniforms (seed 123, as in the notebook)
rng = np.random.default_rng(123); ua = rng.random(10); us = rng.random(10)
print("ua", np.round(ua[:5], 3), "ia", np.round(-np.log(ua[:5]) / 2, 3))
print("us", np.round(us[:5], 3), "sv", np.round(-np.log(us[:5]) / 3, 3))
