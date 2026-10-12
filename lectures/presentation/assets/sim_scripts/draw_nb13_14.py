import os, pickle as pkl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
SP = r"C:\Users\Eshel\AppData\Local\Temp\claude\C--Users-Eshel-workspace-sim-course\1db889cb-ab5e-41fe-b9a0-2a3195bb46a9\scratchpad"
P = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets"
NAVY, BLUE, GREEN, ORANGE, RED, GREY, PURPLE = "#1F3A5F", "#2471A3", "#1E8449", "#D35400", "#C0392B", "#5D6D7E", "#8E44AD"
HIST = "#AFC8DD"
plt.rcParams.update({"font.size": 14})
o = pkl.load(open(os.path.join(SP, "nb13_14.pkl"), "rb"))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def save(fig, deck, name):
    fig.savefig(os.path.join(P, deck, name), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------- NB13 ----------------
A = o["A"]; n5 = A["n5"]; ex = A["exact"]
K = 14; k = np.arange(K)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(k, np.bincount(n5, minlength=K)[:K] / len(n5), color=HIST, label="5,000 simulations")
ax.plot(k, ex[:K], "o", color=NAVY, ms=7, label="exact")
ax.axvspan(10.5, K - 0.5, color=RED, alpha=0.08)
ax.text(12, 0.12, f"P(N ≥ 11)\nsim {(n5 >= 11).mean():.4f}\nexact {ex[11:].sum():.4f}", ha="center", fontsize=13, color=RED)
ax.set_xticks(k); ax.set_xlabel("number of customers at t = 5"); ax.set_ylabel("probability")
ax.legend(frameon=False); clean(ax)
save(fig, "nb13", "pmf_t5.png")

fig, ax = plt.subplots(figsize=(10, 3.8))
h = np.arange(0, 24.01, 0.01); lam = np.where(h < 12, 1, np.where(h < 20, 0.5, 0.9))
ax.plot(h, lam, color=BLUE, lw=3, label="arrival rate λ(t)")
ax.axhline(1, color=NAVY, lw=2, ls="--", label="service rate μ = 1")
ax.fill_between(h, 0, lam, where=h < 12, color=RED, alpha=0.08)
ax.text(6, 0.35, "λ = μ: load 100%", ha="center", fontsize=13, color=RED)
ax.text(16, 0.25, "load 50%", ha="center", fontsize=13, color=GREEN)
ax.text(22, 0.35, "90%", ha="center", fontsize=13, color=ORANGE)
ax.set_xlim(0, 24); ax.set_ylim(0, 1.45); ax.set_xticks(range(0, 25, 2))
ax.set_xlabel("hour of the day"); ax.set_ylabel("jobs per hour")
ax.legend(frameon=False, loc="upper center", ncol=2); clean(ax)
save(fig, "nb13", "schedule.png")

B = o["B"]; base = B["original"]
fig, ax = plt.subplots(figsize=(10, 4.6))
cols = [NAVY, BLUE, GREEN, ORANGE, RED, PURPLE]
for kk in range(6):
    ax.plot(range(25), base["pmf"][:, kk], "o-", color=cols[kk], lw=2.2, ms=4, label=f"{kk} customers")
ax.set_xticks(range(0, 25, 2)); ax.set_xlabel("hour of the day"); ax.set_ylabel("probability")
ax.legend(frameon=False, ncol=3); clean(ax)
save(fig, "nb13", "pmf_over_time.png")

fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(range(25), base["EN"], "o-", color=NAVY, lw=2.6, label="simulation (10,000 runs)")
ax.plot(range(25), base["EN_exact"], "--", color=ORANGE, lw=2, label="exact")
ax.axvspan(0, 12, color=RED, alpha=0.06); ax.axvspan(20, 24, color=ORANGE, alpha=0.06)
ax.set_xticks(range(0, 25, 2)); ax.set_xlabel("hour of the day"); ax.set_ylabel("expected number in system")
ax.legend(frameon=False); clean(ax)
save(fig, "nb13", "EN_over_time.png")

names = ["original", "00-02", "04-06", "10-12", "12-14", "20-22"]
oc = {"original": GREY, "00-02": BLUE, "04-06": GREEN, "10-12": ORANGE, "12-14": RED, "20-22": PURPLE}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for nm in names:
    ax.plot(range(25), B[nm]["EN"], lw=3 if nm in ("04-06", "original") else 1.8, color=oc[nm],
            ls="--" if nm == "original" else "-", label=("no change" if nm == "original" else f"double μ {nm}"))
ax.set_xticks(range(0, 25, 2)); ax.set_xlabel("hour of the day"); ax.set_ylabel("expected number in system")
ax.legend(frameon=False, ncol=2, fontsize=12); clean(ax)
save(fig, "nb13", "options_curves.png")

fig, ax = plt.subplots(figsize=(10, 4.2))
avg = [B[nm]["EN"].mean() for nm in names]
bars = ax.bar(["no change"] + names[1:], avg, color=[GREY] + [GREEN if nm == "04-06" else HIST for nm in names[1:]])
for b, v in zip(bars, avg):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=13, color=NAVY)
ax.set_ylabel("daily average of E[N(t)]"); ax.set_xlabel("hours with doubled service rate"); ax.set_ylim(0, 2.6); clean(ax)
save(fig, "nb13", "options_bar.png")


# ---------------- NB14 ----------------
def line_diagram(name, labels, first="arrivals"):
    fig, ax = plt.subplots(figsize=(11, 2.4)); ax.set_aspect("equal"); ax.axis("off")
    xs = [2.6, 5.6, 8.6]
    if first == "arrivals":
        ax.text(0.2, 0, "Poisson\narrivals\n1 per 20 min", ha="center", va="center", fontsize=13, color=BLUE)
    else:
        ax.add_patch(FancyBboxPatch((-0.75, -0.45), 1.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=GREY, lw=2.2))
        ax.text(0.2, 0, "unlimited\nraw material", ha="center", va="center", fontsize=12, color=GREY, fontweight="bold")
    ax.add_patch(FancyArrowPatch((1.2, 0), (xs[0] - 1.0, 0), arrowstyle="-|>", mutation_scale=20, lw=2.2, color=NAVY))
    for i, (x, lab) in enumerate(zip(xs, labels)):
        ax.add_patch(FancyBboxPatch((x - 0.95, -0.42), 1.9, 0.84, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=NAVY, lw=2.6))
        ax.text(x, 0.06, f"Station {i + 1}", ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
        ax.text(x, -0.75, lab, ha="center", va="center", fontsize=13, color=GREY)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + 1.0, 0), (xs[i + 1] - 1.0, 0), arrowstyle="-|>", mutation_scale=20, lw=2.2, color=NAVY))
    ax.add_patch(FancyArrowPatch((xs[-1] + 1.0, 0), (10.6, 0), arrowstyle="-|>", mutation_scale=20, lw=2.2, color=GREEN))
    ax.text(10.75, 0, "done", va="center", fontsize=14, color=GREEN)
    ax.set_xlim(-0.9, 11.6); ax.set_ylim(-1.05, 0.7)
    save(fig, "nb14", name)


line_diagram("line_poisson.png", ["mean 10 min", "mean 15 min", "mean 60 min"])
line_diagram("line_saturated.png", ["mean 10 min", "mean 45 min", "mean 55 min"], first="saturated")
line_diagram("line_machines.png", ["mean 22 min", "mean 25 min", "mean 20 min"], first="saturated")

D = o["D"]; t = np.arange(241)
sc = {0: BLUE, 1: GREEN, 2: RED}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for s in range(3):
    ax.plot(t, D[(1, 2, 3)]["EN"][:, s], color=sc[s], lw=2.6, label=f"station {s + 1}: servers (1,2,3)")
    ax.plot(t, D[(2, 2, 2)]["EN"][:, s], color=sc[s], lw=2, ls="--", label=f"station {s + 1}: servers (2,2,2)")
ax.set_xlabel("minutes since opening"); ax.set_ylabel("expected number at station")
ax.legend(frameon=False, ncol=2, fontsize=11); clean(ax)
save(fig, "nb14", "tandem_EN.png")

fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.plot(t, D[(1, 2, 3)]["P6"][:, 2], color=RED, lw=2.8, label="station 3 · servers (1,2,3)")
ax.plot(t, D[(2, 2, 2)]["P6"][:, 2], color=RED, lw=2.2, ls="--", label="station 3 · servers (2,2,2)")
for s in (0, 1):
    ax.plot(t, D[(2, 2, 2)]["P6"][:, s], color=sc[s], lw=1.4, ls="--", label=f"station {s + 1} · servers (2,2,2)")
ax.set_xlabel("minutes since opening"); ax.set_ylabel("P(more than 5 at station)")
ax.legend(frameon=False, fontsize=11); clean(ax)
save(fig, "nb14", "tandem_P6.png")

E = o["E"]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bins = np.arange(0, 46) - 0.5
for caps, c in [((2, 3, 3), GREEN), ((3, 3, 2), ORANGE)]:
    ax.hist(E[caps], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=c, edgecolor=c, lw=1.8,
            label=f"servers {caps}: mean {E[caps].mean():.1f}")
ax.set_ylim(0, 0.14)
ax.axvline(30.5, color=RED, ls="--", lw=1.6)
ax.text(31, ax.get_ylim()[1] * 0.85, "more than 30", color=RED, fontsize=12)
ax.set_xlabel("products completed in an 8-hour day"); ax.set_ylabel("probability"); ax.legend(frameon=False, loc="upper left"); clean(ax)
save(fig, "nb14", "throughput_hist.png")

# profit chart for the nine machine scenarios
F = o["F"]; pA, pB, cA, cB = 0.05, 0.07, 10, 8
sc9 = [((2, 2, 2), "AAB"), ((2, 2, 2), "AAA"), ((2, 2, 2), "BBB"), ((2, 2, 2), "ABB"), ((2, 2, 1), "AAA"),
       ((2, 2, 1), "BBB"), ((2, 2, 3), "AAA"), ((2, 2, 3), "BBB"), ((1, 2, 1), "BBB")]
rows = []
for caps, ty in sc9:
    q = np.prod([1 - (pA if c == "A" else pB) for c in ty])
    cost = sum(ci * (cA if c == "A" else cB) for ci, c in zip(caps, ty))
    en = F[caps].mean()
    rows.append((f"{caps} {ty}", 12 * en * q - cost))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(10, 4.8))
labs = [r[0].replace("(", "").replace(")", "").replace(", ", ",") for r in rows]
vals = [r[1] for r in rows]
ax.barh(labs, vals, color=[GREEN if i == len(rows) - 1 else HIST for i in range(len(rows))])
for y, v in enumerate(vals):
    ax.text(v + 2, y, f"{v:.1f}", va="center", fontsize=12, color=NAVY)
ax.set_xlabel("expected daily profit"); ax.set_xlim(0, 290)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, axis="x")
save(fig, "nb14", "profit_bar.png")
print("ok")
