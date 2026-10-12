import os, pickle as pkl
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
ROOT = r"C:\Users\Eshel\workspace\sim_course"
OUT = ROOT + r"\lectures\presentation\assets\nb11"
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
NAVY, BLUE, GREEN, ORANGE, RED, GREY, PURPLE = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#C0392B", "#5D6D7E", "#8E44AD"
HIST = "#AFC8DD"
plt.rcParams.update({"font.size": 14})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight"); plt.close(fig)
def ph_pdf(x, a, S):
    a = np.asarray(a).ravel(); S = np.asarray(S); s = -S.sum(axis=1)
    return np.array([a @ expm(S * xi) @ s for xi in x])

a, S, data = pkl.load(open(ROOT + r"\pkls\fitting_example.pkl", "rb"))
em = pkl.load(open(os.path.join(SP, "nb11_em.pkl"), "rb"))
x, ytrue = np.load(os.path.join(OUT, "_true_pdf.npy"))

# EM fits 2x2
fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
for ax, k in zip(axs.ravel(), sorted(em)):
    ax.hist(data, bins=250, density=True, color=HIST)
    ax.plot(x, ytrue, color=NAVY, lw=2, ls="--")
    ax.plot(x, ph_pdf(x, em[k][0], em[k][1]), color=ORANGE, lw=2.8)
    ax.set_title(f"PH with {k} phases", fontsize=15); ax.set_xlim(0, 3); clean(ax)
axs[0, 0].plot([], [], color=ORANGE, lw=2.8, label="EM fit"); axs[0, 0].plot([], [], color=NAVY, lw=2, ls="--", label="true density")
axs[0, 0].legend(frameon=False, fontsize=12)
for ax in axs[1]: ax.set_xlabel("service time")
fig.tight_layout(); save(fig, "em_fits.png")

# runtime
ks = sorted(em); rt = [em[k][2] for k in ks]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(ks, rt, "o-", color=NAVY, lw=2.6, ms=8)
for k, r in zip(ks, rt): ax.annotate(f"{r:,.0f} s" if r >= 10 else f"{r:.1f} s", (k, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=13, color=NAVY)
ax.set_xticks(ks); ax.set_xlabel("number of phases"); ax.set_ylabel("EM runtime (seconds)"); clean(ax)
ax.set_ylim(0, max(rt) * 1.18)
save(fig, "em_runtime.png")

# queue pmf comparison
q = pkl.load(open(os.path.join(SP, "nb11_queue.pkl"), "rb"))
K = 21; n = np.arange(K)
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(n, q["true"]["pmf"][:K], color=HIST, label="true distribution", width=0.85)
for name, c, mk in [("exponential", RED, "s"), ("PH-5", ORANGE, "^"), ("PH-20", GREEN, "o")]:
    ax.plot(n, q[name]["pmf"][:K], mk + "-", color=c, lw=2, ms=6, label=name.replace("PH-", "PH, ") + (" phases" if "PH" in name else " fit"))
ax.set_xticks(n[::2]); ax.set_xlabel("number of customers in the system"); ax.set_ylabel("long-run fraction of time")
ax.legend(frameon=False); clean(ax)
save(fig, "queue_pmf.png")
print("ok", [(k, round(em[k][2], 1)) for k in ks])
