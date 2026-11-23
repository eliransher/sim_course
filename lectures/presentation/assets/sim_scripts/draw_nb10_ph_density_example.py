"""Worked numerical example for deck 10: density at a point and mean of a PH(alpha, S).

Hyper-exponential of size 2 (example 2 of the deck): alpha = (0.8, 0.2), S = diag(-4, -1/4).
  f(1) = alpha e^{S} s   = 0.8*4*e^-4 + 0.2*0.25*e^-0.25 = 0.059 + 0.039 = 0.098
  E[X] = alpha (-S)^-1 1 = 0.8/4 + 0.2*4 = 1
Writes assets/nb10/ph_density_example.png.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv

OUT = r"C:\Users\Eshel\workspace\sim_course\lectures\presentation\assets\nb10"
NAVY, GREEN, VIOLET, GREY = "#1F3A5F", "#1E8449", "#7C3AED", "#5D6D7E"
FS = 17

# ---- the numbers (checked, not typed by hand)
a = np.array([0.8, 0.2]); S = np.diag([-4.0, -0.25]); s = -S @ np.ones(2); x = 1.0
f1 = a @ expm(S * x) @ s
terms = a * np.exp(np.diag(S) * x) * s
mean = a @ inv(-S) @ np.ones(2)
assert abs(terms.sum() - f1) < 1e-12 and abs(mean - 1) < 1e-12
t1, t2 = (round(v, 3) for v in terms)
F1 = round(f1, 3)
assert round(t1 + t2, 3) == F1, (t1, t2, F1)

fig = plt.figure(figsize=(12.5, 4.5))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.5); ax.axis("off")
R = fig.canvas.get_renderer()


def width(t, fs=FS):
    art = ax.text(0, 0, t, fontsize=fs); bb = art.get_window_extent(R); art.remove()
    return bb.width / fig.dpi  # 1 data unit = 1 inch


def text(x, y, t, color=NAVY, fs=FS, gap=0.12):
    ax.text(x, y, t, ha="left", va="center", fontsize=fs, color=color)
    return x + width(t, fs) + gap


def mat(x, y, rows, color=NAVY, cw=None, ch=0.42, fs=FS, gap=0.14):
    """Bracketed matrix centred vertically on y; returns right edge."""
    nr, nc = len(rows), len(rows[0])
    cw = cw or max(width(v, fs) for r in rows for v in r) + 0.28
    h = nr * ch; top = y + h / 2
    x0 = x + 0.04; x1 = x0 + nc * cw + 0.08
    for br, d in ((x0, 0.08), (x1, -0.08)):
        ax.plot([br + d, br, br, br + d], [top + 0.03, top + 0.03, top - h - 0.03, top - h - 0.03], color=color, lw=1.7)
    for i, r in enumerate(rows):
        for k, v in enumerate(r):
            ax.text(x0 + 0.04 + (k + 0.5) * cw, top - (i + 0.5) * ch, v, ha="center", va="center", fontsize=fs, color=color)
    return x1 + gap


def label(y, t):
    ax.text(0.15, y, t, ha="left", va="center", fontsize=13, color=GREY, style="italic")


X0 = 2.0
# ---- row 1: the data
y = 3.85
label(y, "the data")
p = text(X0, y, r"$\alpha =$")
p = mat(p, y, [["0.8", "0.2"]])
p = text(p + 0.35, y, r"$S =$")
p = mat(p, y, [["$-4$", "0"], ["0", "$-0.25$"]])
p = text(p + 0.35, y, r"$s = -S\mathbf{1} =$")
p = mat(p, y, [["4"], ["0.25"]])

# ---- row 2: density at x = 1
y = 2.3
label(y, "density at x = 1")
p = text(X0, y, r"$f(1) = \alpha\, e^{S \cdot 1}\, s =$")
p = mat(p, y, [["0.8", "0.2"]])
p = mat(p, y, [["$e^{-4}$", "0"], ["0", "$e^{-0.25}$"]])
p = mat(p, y, [["4"], ["0.25"]])
y2 = 1.62
p = text(X0, y2, r"$= 0.8 \cdot e^{-4} \cdot 4 \;+\; 0.2 \cdot e^{-0.25} \cdot 0.25 \;=\; %.3f + %.3f \;\approx\; %.3f$" % (t1, t2, F1),
         color=VIOLET)

# ---- row 3: the mean
y = 0.55
label(y, "mean")
p = text(X0, y, r"$E[X] = \alpha\, (-S)^{-1}\, \mathbf{1} =$")
p = mat(p, y, [["0.8", "0.2"]])
p = mat(p, y, [["0.25", "0"], ["0", "4"]])
p = mat(p, y, [["1"], ["1"]])
p = text(p, y, r"$= 0.8 \cdot 0.25 + 0.2 \cdot 4 = 0.2 + 0.8 = 1$", color=GREEN)

ax.plot([0.15, 12.3], [1.15, 1.15], color="#E5E7EB", lw=1)
ax.plot([0.15, 12.3], [3.15, 3.15], color="#E5E7EB", lw=1)
fig.savefig(os.path.join(OUT, "ph_density_example.png"), dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.1)
print("f(1) =", f1, "terms", terms, "mean", mean)
