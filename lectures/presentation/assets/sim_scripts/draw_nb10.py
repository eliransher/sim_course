import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.stats import gamma
OUT = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets\nb10"
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
NAVY, BLUE, GREEN, ORANGE, RED, GREY = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#C0392B", "#5D6D7E"
plt.rcParams.update({"font.size": 14})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)
def hyper_pdf(x, p, r): return sum(pi*ri*np.exp(-ri*x) for pi, ri in zip(p, r))

# 1. three densities, same mean 1
x = np.linspace(0.001, 4, 600)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(x, gamma.pdf(x, 5, scale=0.2), color=GREEN, lw=3, label="Erlang (5 phases)   SCV = 0.2")
ax.plot(x, np.exp(-x), color=BLUE, lw=3, label="Exponential            SCV = 1")
ax.plot(x, hyper_pdf(x, [0.05, 0.95], [0.1, 1.9]), color=RED, lw=3, label="Hyper-exponential   SCV = 9.5")
ax.set_xlabel("x"); ax.set_ylabel("density"); ax.set_ylim(0, 2); ax.legend(frameon=False); clean(ax)
ax.set_title("Three distributions, all with mean 1", fontsize=15)
save(fig, "three_densities.png")

# 2. variance paradox: Erlang(5, rate 1) vs hyper(0.5,0.5; 0.5,10)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
x = np.linspace(0.001, 14, 700)
axs[0].plot(x, gamma.pdf(x, 5, scale=1), color=GREEN, lw=3)
axs[0].set_title("Erlang:  E[X] = 5,  V[X] = 5", fontsize=15)
x2 = np.linspace(0.001, 6, 700)
axs[1].plot(x2, hyper_pdf(x2, [0.5, 0.5], [0.5, 10]), color=RED, lw=3)
axs[1].set_title("Hyper-exponential:  E[X] = 1.05,  V[X] = 2.91", fontsize=15)
for a in axs: a.set_xlabel("x"); clean(a)
axs[0].set_ylabel("density")
save(fig, "variance_paradox.png")

# 3. phase diagrams
def circ(ax, x, y, t, c=NAVY, r=0.42):
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=2.6, zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=17, fontweight="bold", color=c, zorder=4)
def arr(ax, p, q, c=NAVY, lab=None, lxy=None, fs=16):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, lw=2.2, color=c, zorder=2))
    if lab: ax.text(*lxy, lab, ha="center", va="center", fontsize=fs, color=c)
fig, ax = plt.subplots(figsize=(11, 2.2)); ax.set_aspect("equal"); ax.axis("off")
xs = [1.5 * i for i in range(5)]
arr(ax, (-1.4, 0), (-0.42, 0), GREY, "start", (-1.0, 0.35), fs=14)
for i, xx in enumerate(xs):
    circ(ax, xx, 0, str(i + 1))
    nxt = (xx + 1.5 - 0.42, 0) if i < 4 else (xx + 1.5, 0)
    arr(ax, (xx + 0.42, 0), nxt, GREEN if i == 4 else NAVY, r"$\mu$", (xx + 0.75, 0.33))
ax.text(xs[-1] + 1.95, 0, "done", va="center", fontsize=15, color=GREEN)
ax.set_xlim(-1.6, 8.9); ax.set_ylim(-0.6, 0.7); save(fig, "phases_erlang.png")

fig, ax = plt.subplots(figsize=(8, 3.2)); ax.set_aspect("equal"); ax.axis("off")
ax.plot(0, 0, "o", color=GREY, ms=10)
ax.text(-0.25, 0, "start", ha="right", va="center", fontsize=14, color=GREY)
circ(ax, 2.2, 0.9, "1"); circ(ax, 2.2, -0.9, "2")
arr(ax, (0.1, 0.05), (1.8, 0.8), GREY, r"$p_1$", (0.8, 0.75))
arr(ax, (0.1, -0.05), (1.8, -0.8), GREY, r"$p_2$", (0.8, -0.75))
arr(ax, (2.62, 0.85), (4.3, 0.1), GREEN, r"$\mu_1$", (3.6, 0.85))
arr(ax, (2.62, -0.85), (4.3, -0.1), GREEN, r"$\mu_2$", (3.6, -0.85))
ax.text(4.45, 0, "done", va="center", fontsize=15, color=GREEN)
ax.set_xlim(-1.2, 5.3); ax.set_ylim(-1.5, 1.5); save(fig, "phases_hyper.png")

# 4. random PH densities (notebook's sampler, seeded)
sys.path[:0] = [r"C:\Users\Eshel\workspace\sim_course\code", r"C:\Users\Eshel\workspace\sim_course\code\butools2\Python"]
cwd = os.getcwd(); os.chdir(r"C:\Users\Eshel\workspace\sim_course\code")
import sample_PH as sp
os.chdir(cwd)
np.random.seed(int(os.environ.get("PHSEED", "3")))
fig, ax = plt.subplots(figsize=(10, 4.6))
xv = np.linspace(0, 2, 200)
cols = [NAVY, GREEN, ORANGE, RED, "#8E44AD"]
for k in range(5):
    a, A, moms, _ = sp.sample(1, 150)
    yv = np.asarray(sp.compute_pdf_within_range(xv, a, A)).ravel()
    m1, m2 = float(np.ravel(moms[0])[0]), float(np.ravel(moms[1])[0])
    ax.plot(xv, yv, lw=2.6, color=cols[k], label=f"SCV = {(m2 - m1**2)/m1**2:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend(frameon=False); clean(ax)
ax.set_title("Five randomly generated PH distributions", fontsize=15)
save(fig, "random_ph.png")

# 5. queue-length pmf comparison
pm = np.load(os.path.join(SP, "nb10_pmf.npy"), allow_pickle=True).item()
K = 16
fig, ax = plt.subplots(figsize=(11, 4.6))
w = 0.4; n = np.arange(K)
ax.bar(n - w/2, pm["erlang"][:K], w, color=GREEN, label="Erlang service (SCV 0.2)")
ax.bar(n + w/2, pm["hyper"][:K], w, color=RED, label="Hyper-exponential service (SCV 9.5)")
th, te = pm["hyper"][K:].sum(), pm["erlang"][K:].sum()
ax.text(K - 0.6, 0.2, f"P(N ≥ {K}):\nhyper {th:.3f}\nErlang {te:.4f}", ha="right", fontsize=13, color=NAVY)
ax.set_xticks(n); ax.set_xlabel("number of customers in the system"); ax.set_ylabel("long-run fraction of time")
ax.legend(frameon=False); clean(ax)
save(fig, "queue_pmf.png")
print("tail", th, te)
