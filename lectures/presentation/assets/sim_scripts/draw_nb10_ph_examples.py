"""Figures for the PH-examples slides added to deck 10 (all examples have mean 1).

Erlang(4, rate 4); hyper-exponential alpha=(0.8,0.2), rates (4, 0.25);
hyper-Erlang: w.p. 1/4 Erlang(4, rate 16), w.p. 3/4 Erlang(5, rate 4).
"""
import os
from math import factorial
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.linalg import expm, inv

OUT = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets\nb10"
NAVY, GREEN, RED, VIOLET, GREY = "#1F3A5F", "#1E8449", "#C0392B", "#7C3AED", "#5D6D7E"
plt.rcParams.update({"font.size": 14})


# ---------------------------------------------------------------- PH helpers
def erlang_T(k, r):
    return np.diag([-r] * k) + np.diag([r] * (k - 1), 1)


def block(*Ts):
    n = sum(t.shape[0] for t in Ts)
    M = np.zeros((n, n)); i = 0
    for t in Ts:
        k = t.shape[0]; M[i:i + k, i:i + k] = t; i += k
    return M


def moment(a, T, n):
    return factorial(n) * a @ np.linalg.matrix_power(inv(-T), n) @ np.ones(len(a))


def stats(a, T):
    m1, m2, m3 = (moment(a, T, n) for n in (1, 2, 3))
    v = m2 - m1 ** 2
    return dict(mean=m1, var=v, scv=v / m1 ** 2, skew=(m3 - 3 * m1 * v - m1 ** 3) / v ** 1.5)


def pdf(a, T, x):
    s = -T @ np.ones(len(a))
    return np.array([a @ expm(T * xx) @ s for xx in x])


EXAMPLES = {
    "erlang": (np.array([1., 0, 0, 0]), erlang_T(4, 4.0)),
    "hyper": (np.array([0.8, 0.2]), np.diag([-4.0, -0.25])),
}
_a = np.zeros(9); _a[0] = 0.25; _a[4] = 0.75
EXAMPLES["hyerl"] = (_a, block(erlang_T(4, 16.0), erlang_T(5, 4.0)))


# ---------------------------------------------------------------- drawing helpers
def circ(ax, x, y, t, c=NAVY, r=0.36, fs=15):
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=2.4, zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold", color=c, zorder=4)


def arr(ax, p, q, c=NAVY, lab=None, lxy=None, fs=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c, zorder=2))
    if lab:
        ax.text(*lxy, lab, ha="center", va="center", fontsize=fs, color=c)


def matrix(ax, x, y, label, rows, cw=0.62, ch=0.42, fs=14, color=NAVY):
    """Draw `label = [rows]` with its top-left text corner at (x, y). Returns right edge."""
    nr, nc = len(rows), len(rows[0])
    h = nr * ch
    ax.text(x, y - h / 2, label, ha="right", va="center", fontsize=fs + 3, color=color)
    x0 = x + 0.18
    for j, br in enumerate((x0, x0 + nc * cw + 0.1)):
        d = 0.09 if j == 0 else -0.09
        ax.plot([br + d, br, br, br + d], [y + 0.04, y + 0.04, y - h - 0.04, y - h - 0.04], color=color, lw=1.8)
    for i, row in enumerate(rows):
        for k, v in enumerate(row):
            ax.text(x0 + 0.05 + (k + 0.5) * cw, y - (i + 0.5) * ch, v, ha="center", va="center",
                    fontsize=fs, color=color)
    return x0 + nc * cw + 0.1


def fmt(v):
    if v == 0:
        return "0"
    s = f"{abs(v):g}"
    return ("−" if v < 0 else "") + s


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1. Erlang structure
xs = [1.2 * i for i in range(4)]
a, T = EXAMPLES["erlang"]

# redo with a clean layout: diagram left, alpha on top-right, T below it
def erlang_fig():
    fig, ax = plt.subplots(figsize=(14, 3.0)); ax.set_aspect("equal"); ax.axis("off")
    arr(ax, (-1.2, 0), (-0.36, 0), GREY, "start", (-0.85, 0.32), fs=13)
    for i, xx in enumerate(xs):
        circ(ax, xx, 0, str(i + 1))
        nxt = (xx + 1.2 - 0.36, 0) if i < 3 else (xx + 1.25, 0)
        arr(ax, (xx + 0.36, 0), nxt, GREEN if i == 3 else NAVY, "4", (xx + 0.6, 0.27))
    ax.text(xs[-1] + 1.35, 0, "done", va="center", fontsize=14, color=GREEN)
    matrix(ax, 8.0, 0.98, r"$\alpha$ =", [[fmt(v) for v in a]], cw=0.7)
    matrix(ax, 8.0, 0.18, r"$T$ =", [[fmt(v) for v in r] for r in T], cw=0.7, ch=0.36)
    ax.set_xlim(-1.5, 11.2); ax.set_ylim(-1.35, 1.15)
    save(fig, "ph_ex_erlang.png")

erlang_fig()

# ---------------------------------------------------------------- 2. hyper-exponential structure
fig, ax = plt.subplots(figsize=(14, 3.0)); ax.set_aspect("equal"); ax.axis("off")
ax.plot(0, 0, "o", color=GREY, ms=9)
ax.text(-0.22, 0, "start", ha="right", va="center", fontsize=13, color=GREY)
circ(ax, 2.0, 0.75, "1"); circ(ax, 2.0, -0.75, "2")
arr(ax, (0.1, 0.05), (1.66, 0.66), GREY, "0.8", (0.75, 0.62))
arr(ax, (0.1, -0.05), (1.66, -0.66), GREY, "0.2", (0.75, -0.62))
arr(ax, (2.36, 0.7), (3.9, 0.08), GREEN, "4", (3.25, 0.68))
arr(ax, (2.36, -0.7), (3.9, -0.08), GREEN, "0.25", (3.3, -0.7))
ax.text(4.05, 0, "done", va="center", fontsize=14, color=GREEN)
a, T = EXAMPLES["hyper"]
matrix(ax, 7.6, 0.75, r"$\alpha$ =", [[fmt(v) for v in a]], cw=0.95)
matrix(ax, 7.6, 0.0, r"$T$ =", [[fmt(v) for v in r] for r in T], cw=0.95)
ax.set_xlim(-1.3, 10.5); ax.set_ylim(-1.2, 1.2)
save(fig, "ph_ex_hyper.png")

# ---------------------------------------------------------------- 3. hyper-Erlang structure
fig, ax = plt.subplots(figsize=(14, 3.8)); ax.set_aspect("equal"); ax.axis("off")
ax.plot(0, 0, "o", color=GREY, ms=9)
ax.text(-0.22, 0, "start", ha="right", va="center", fontsize=13, color=GREY)
yt, yb, sp, R = 0.95, -0.95, 1.05, 0.3
top = [1.7 + sp * i for i in range(4)]
bot = [1.7 + sp * i for i in range(5)]
arr(ax, (0.1, 0.06), (top[0] - R - 0.03, yt - 0.12), GREY, "¼", (0.62, 0.72), fs=16)
arr(ax, (0.1, -0.06), (bot[0] - R - 0.03, yb + 0.12), GREY, "¾", (0.62, -0.72), fs=16)
end_x = bot[-1] + 1.15
for i, xx in enumerate(top):
    circ(ax, xx, yt, str(i + 1), r=R, fs=13)
    if i < 3:
        arr(ax, (xx + R, yt), (xx + sp - R, yt), NAVY, "16", (xx + sp / 2, yt + 0.3), fs=12)
arr(ax, (top[-1] + R, yt), (end_x - 0.05, 0.12), GREEN, "16", (top[-1] + 1.05, yt + 0.12), fs=12)
for i, xx in enumerate(bot):
    circ(ax, xx, yb, str(i + 5), r=R, fs=13)
    if i < 4:
        arr(ax, (xx + R, yb), (xx + sp - R, yb), NAVY, "4", (xx + sp / 2, yb + 0.3), fs=12)
arr(ax, (bot[-1] + R, yb), (end_x - 0.05, -0.12), GREEN, "4", (bot[-1] + 0.75, yb + 0.05), fs=12)
ax.text(end_x + 0.08, 0, "done", va="center", fontsize=14, color=GREEN)
ax.text((top[0] + top[-1]) / 2, yt + 0.7, "4 phases, rate 16", ha="center", fontsize=12, color=GREY)
ax.text((bot[0] + bot[-1]) / 2, yb - 0.72, "5 phases, rate 4", ha="center", fontsize=12, color=GREY)
# matrices: alpha in full, T in block form, both blocks in full
X = 10.0
matrix(ax, X, 1.85, r"$\alpha$ =", [["¼", "0", "0", "0", "¾", "0", "0", "0", "0"]], cw=0.42, fs=13)
matrix(ax, X, 0.55, r"$T$ =", [[r"$T_1$", "0"], ["0", r"$T_2$"]], cw=0.62, ch=0.42, fs=14)
X2 = X + 2.9
r1 = [["−16", "16", "0", "0"], ["0", "−16", "16", "0"], ["0", "0", "−16", "16"], ["0", "0", "0", "−16"]]
matrix(ax, X2, 1.25, r"$T_1$ =", r1, cw=0.6, ch=0.34, fs=11)
r2 = [["−4", "4", "0", "0", "0"], ["0", "−4", "4", "0", "0"], ["0", "0", "−4", "4", "0"],
      ["0", "0", "0", "−4", "4"], ["0", "0", "0", "0", "−4"]]
matrix(ax, X2, -0.5, r"$T_2$ =", r2, cw=0.5, ch=0.34, fs=11)
ax.set_xlim(-1.3, 16.1); ax.set_ylim(-2.4, 2.15)
save(fig, "ph_ex_hyerl.png")

# ---------------------------------------------------------------- 4. densities
x = np.linspace(0.002, 3.5, 500)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(x, np.exp(-x), color=GREY, lw=2, ls="--", label="Exponential (reference)")
lab = {"erlang": ("Erlang (4 phases)", GREEN), "hyper": ("Hyper-exponential", RED),
       "hyerl": ("Hyper-Erlang (two modes)", VIOLET)}
for k, (name, c) in lab.items():
    a, T = EXAMPLES[k]; st = stats(a, T)
    ax.plot(x, pdf(a, T, x), color=c, lw=3,
            label=f"{name}   SCV = {st['scv']:.2f},  skew = {st['skew']:.2f}")
ax.axvline(1, color=NAVY, lw=1, ls=":")
ax.text(1.03, 2.35, "mean = 1", color=NAVY, fontsize=12)
ax.set_ylim(0, 2.5); ax.set_xlim(0, 3.5)
ax.set_xlabel("x"); ax.set_ylabel("density")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=12)
ax.set_title("Three PH distributions, all with mean 1", fontsize=15)
save(fig, "ph_ex_pdfs.png")

# ---------------------------------------------------------------- 5. skewness illustration
from scipy.stats import gamma
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
fig.suptitle(r"$\gamma \;=\; \frac{E[(X-E[X])^3]}{\sigma^3}$", fontsize=24, color=NAVY, y=0.98)
xx = np.linspace(0, 10, 400)
for ax_, y, t, c in [
        (axs[0], gamma.pdf(10 - xx, 2.2, scale=1.3), "negative skew: long left tail", RED),
        (axs[1], np.exp(-(xx - 5) ** 2 / 2.5), "zero skew: symmetric", NAVY),
        (axs[2], gamma.pdf(xx, 2.2, scale=1.3), "positive skew: long right tail", GREEN)]:
    y = y / y.max()
    ax_.fill_between(xx, y, color=c, alpha=0.18); ax_.plot(xx, y, color=c, lw=2.6)
    ax_.set_title(t, fontsize=13, color=c); ax_.set_yticks([]); ax_.set_xticks([])
    for s in ("top", "right", "left"): ax_.spines[s].set_visible(False)
fig.subplots_adjust(top=0.70, bottom=0.04, left=0.02, right=0.98, wspace=0.08)
save(fig, "skewness_shapes.png")

for k in EXAMPLES:
    print(k, {n: round(v, 4) for n, v in stats(*EXAMPLES[k]).items()})

# ---------------------------------------------------------------- 6. jump probabilities: 3-phase PH with feedback
# T = [[-4, 2, 1], [0, -2, 1], [1, 0, -3]],  t = -T1 = (1, 1, 2),  alpha = (1, 0, 0)
from matplotlib.patches import FancyArrowPatch as FAP
fig, ax = plt.subplots(figsize=(14, 3.9)); ax.set_aspect("equal"); ax.axis("off")
P1, P2, P3, DN = (1.2, 0), (3.6, 1.1), (3.6, -1.1), (6.4, 0)
R = 0.38
arr(ax, (-0.3, 0), (P1[0] - R, 0), GREY, "start", (0.25, 0.3), fs=13)
for p, lab, rate in [(P1, "1", "exit rate 4"), (P2, "2", "exit rate 2"), (P3, "3", "exit rate 3")]:
    circ(ax, *p, lab, r=R)
    ax.text(p[0], p[1] + (0.62 if p[1] >= 0 else -0.62), rate, ha="center", va="center", fontsize=11, color=GREY)
ax.text(DN[0] + 0.05, 0, "done", va="center", fontsize=15, color=GREEN)
def edge(p, q, lab, lxy, c=NAVY, rad=0.0):
    ax.add_patch(FAP(p, q, arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c,
                     connectionstyle=f"arc3,rad={rad}", zorder=2))
    ax.text(*lxy, lab, ha="center", va="center", fontsize=14, color=c, fontweight="bold")
edge((P1[0] + 0.3, 0.24), (P2[0] - 0.36, P2[1] - 0.1), "½", (2.2, 0.85))
edge((P1[0] + 0.3, -0.24), (P3[0] - 0.36, P3[1] + 0.1), "¼", (2.55, -0.38))
edge((P2[0], P2[1] - R), (P3[0], P3[1] + R), "½", (3.85, 0.45))
edge((P3[0] - 0.33, P3[1] - 0.2), (P1[0] - 0.05, -R), "⅓", (1.75, -1.45), c=VIOLET, rad=-0.35)
edge((P1[0] + R, 0.0), (DN[0] - 0.05, 0.0), "¼", (5.0, 0.18), c=GREEN)
edge((P2[0] + R, P2[1] - 0.1), (DN[0] - 0.05, 0.15), "½", (5.15, 0.85), c=GREEN)
edge((P3[0] + R, P3[1] + 0.1), (DN[0] - 0.05, -0.15), "⅔", (5.15, -0.85), c=GREEN)
# matrices
X = 9.6
matrix(ax, X, 1.55, r"$T$ =", [["−4", "2", "1"], ["0", "−2", "1"], ["1", "0", "−3"]], cw=0.55, ch=0.4, fs=13)
matrix(ax, X, 0.0, r"$t$ =", [["1"], ["1"], ["2"]], cw=0.55, ch=0.4, fs=13)
X2 = X + 3.3
hdr = ["→1", "→2", "→3", "done"]
for k, h in enumerate(hdr):
    ax.text(X2 + 0.23 + (k + 0.5) * 0.62, 1.8, h, ha="center", va="center", fontsize=11,
            color=GREEN if h == "done" else GREY)
matrix(ax, X2, 1.55, "", [["0", "½", "¼", "¼"], ["0", "0", "½", "½"], ["⅓", "0", "0", "⅔"]], cw=0.62, ch=0.4, fs=13)
ax.text(X2 + 0.25 + 2 * 0.62, -0.05, "jump probabilities\n(each row sums to 1)", ha="center", va="top",
        fontsize=11, color=GREY)
ax.set_xlim(-0.6, 16.4); ax.set_ylim(-2.0, 2.1)
save(fig, "ph_jumps.png")
