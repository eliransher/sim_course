# -*- coding: utf-8 -*-
"""Insert two slides on PH jump probabilities (where do we go when a phase ends?)
into deck 10, right after the hyper-Erlang example (slide 14).
Figure: assets/sim_scripts/draw_nb10_ph_examples.py (ph_jumps.png).
Usage: python build_nb10_ph_jumps.py <temp_dir> [output_path]  (default: overwrite the deck)
"""
import os, sys
from pptx import Presentation
from deckbuilder import Deck, make_formula
from deck10_insert import restyle, transplant, renumber_footers

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets", "nb10")
F = os.path.join(A, "formulas")
DECK = os.path.join(HERE, "notebook_10_PH_SCV.pptx")
INSERT_AFTER = 14
TMP = os.path.join(sys.argv[1], "nb10_ph_jumps_tmp.pptx")

d = Deck(footer_label="שיעור 10 · שונות ותורים")

d.figure_text(
    "לאן הולכים כשיוצאים משלב?",
    make_formula(F, "ph_jumps", [
        r"P(i \to j) = \frac{T_{ij}}{-T_{ii}}, \qquad P(i \to \mathrm{done}) = \frac{t_i}{-T_{ii}}, \qquad t = -T\mathbf{1}",
        r"\sum_{j \neq i} P(i \to j) \;+\; P(i \to \mathrm{done}) \;=\; 1",
    ], fontsize=30),
    [
        ("זמן השהייה בשלב", "מעריכי, בקצב השווה למינוס האיבר שעל האלכסון — קצב היציאה הכולל מהשלב."),
        ("מעבר לשלב אחר", "ההסתברות היא קצב המעבר לאותו שלב, חלקי קצב היציאה הכולל."),
        ("סיום — המצב הבולע", "ההסתברות היא קצב הסיום מהשלב, חלקי קצב היציאה הכולל."),
        ("סכום ההסתברויות", "תמיד 1: כל יציאה משלב היא מעבר לשלב אחר, או סיום."),
    ],
    kicker="Jump probabilities  ·  done = absorbing state",
    img_h=1.75,
)

d.figure_text(
    "דוגמה: שלושה שלבים עם חזרה",
    os.path.join(A, "ph_jumps.png"),
    [
        ("משלב 1", "קצב יציאה 4: לשלב 2 בהסתברות חצי, לשלב 3 ברבע, ולסיום ברבע."),
        ("משלבים 2 ו־3", "משלב 2 — חצי לשלב 3 וחצי לסיום. משלב 3 — שליש חזרה לשלב 1, ושני שלישים לסיום."),
        ("בדוגמאות שלנו", "בארלנג ובהיפר־ארלנג עוברים תמיד לשלב הבא, ומהשלב האחרון לסיום. בהיפר־מעריכית מסיימים מיד."),
    ],
    kicker="Example  ·  each row of jump probabilities sums to 1",
    img_h=2.6,
)
d.save(TMP)

src = Presentation(TMP)
restyle(src)
dst = Presentation(DECK)
assert "דוגמה 3" in "".join(sh.text_frame.text for sh in dst.slides[INSERT_AFTER - 1].shapes if sh.has_text_frame)
transplant(src, dst, INSERT_AFTER)
renumber_footers(dst, "שיעור 10")
OUT = sys.argv[2] if len(sys.argv) > 2 else DECK
dst.save(OUT)
print("deck 10 now has", len(dst.slides), "slides")
