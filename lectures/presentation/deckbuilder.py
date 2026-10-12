# -*- coding: utf-8 -*-
"""
Reusable helpers for building the course lecture decks.

Design goals:
  * Hebrew narrative (RTL) with English technical terms kept in English.
  * Clean, modern, consistent look across all lectures.
  * Simple, declarative API so each notebook deck is a short content script.
"""
import re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
NAVY    = RGBColor(0x1F, 0x3A, 0x5F)   # primary / dividers
TEAL    = RGBColor(0x2E, 0x86, 0xAB)   # accent
ORANGE  = RGBColor(0xE0, 0x7A, 0x2F)   # highlight
INK     = RGBColor(0x1A, 0x24, 0x33)   # body text
MUTED   = RGBColor(0x5A, 0x6B, 0x7B)   # secondary text
PANEL   = RGBColor(0xF1, 0xF5, 0xF9)   # card background
PANEL2  = RGBColor(0xE7, 0xEE, 0xF4)   # alt card
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
LINE    = RGBColor(0xD3, 0xDD, 0xE6)

HEBREW_FONT = "Calibri"   # renders Hebrew cleanly on Windows/Office

EMU_W = Inches(13.333)
EMU_H = Inches(7.5)


# ----------------------------------------------------------------------------
# Low-level text helpers
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Bidi isolation: mixing Hebrew with Latin/digits/equations in one string
# causes PowerPoint's bidi algorithm to misorder/mirror neutral characters
# (brackets, =, +, arrows). Wrap every Latin/digit/math span with Unicode
# directional isolate marks (LRI ... PDI) so it renders as one atomic
# left-to-right unit, regardless of the surrounding Hebrew.
# ----------------------------------------------------------------------------
LRI = '‪'  # Left-to-Right Embedding (LRE) — old-style bidi override,
PDI = '‬'  # Pop Directional Formatting (PDF) — wide Office/font support,
                # unlike the newer LRI/PDI isolates (Unicode 6.3) which many
                # fonts render as a missing-glyph box instead of treating as
                # invisible control characters.
_HEB_RE = re.compile(r'[֐-׿]')
# Digits alone don't trigger isolation: Hebrew + plain numbers renders fine
# natively, while LRE/PDF marks around a bare number misplace its spaces.
_LTR_TRIGGER_RE = re.compile(r'[A-Za-zͰ-Ͽ₀-₉²³¹]')


def bidi_isolate(text):
    if not text or not _LTR_TRIGGER_RE.search(text):
        return text  # nothing Latin/numeric in here; leave pure Hebrew alone
    parts = re.split(r'(\s+)', text)
    types = []
    for p in parts:
        if p == '':
            types.append(None)
        elif p.strip() == '':
            types.append('SPACE')
        elif _HEB_RE.search(p):
            types.append('HEB')
        elif _LTR_TRIGGER_RE.search(p):
            types.append('LTR')
        else:
            types.append('NEUT')
    n = len(parts)
    out = []
    i = 0
    while i < n:
        if types[i] in ('LTR', 'NEUT') and parts[i] != '':
            j = i
            has_ltr = False
            while j < n and types[j] in ('LTR', 'NEUT', 'SPACE'):
                if types[j] == 'LTR':
                    has_ltr = True
                j += 1
            k = j
            while k > i and types[k - 1] == 'SPACE':
                k -= 1
            if has_ltr and k > i:
                out.append(LRI + ''.join(parts[i:k]) + PDI)
                out.extend(parts[k:j])
                i = j
                continue
        out.append(parts[i])
        i += 1
    return ''.join(out)


def _set_rtl(paragraph, rtl=True):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set('rtl', '1' if rtl else '0')


def _no_autofit(tf):
    # keep our sizing; disable shrink-to-fit surprises
    try:
        tf.word_wrap = True
    except Exception:
        pass


def set_text(shape, runs, align=PP_ALIGN.RIGHT, rtl=True, anchor=MSO_ANCHOR.TOP,
             space_after=6, line=1.0):
    """runs: list of paragraphs; each paragraph is a list of (text, opts) tuples.
    opts keys: size, bold, color, italic, font."""
    tf = shape.text_frame
    _no_autofit(tf)
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    first = True
    for para in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        _set_rtl(p, rtl)
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        try:
            p.line_spacing = line
        except Exception:
            pass
        for text, opts in para:
            r = p.add_run()
            r.text = bidi_isolate(text)
            f = r.font
            f.name = opts.get('font', HEBREW_FONT)
            f.size = Pt(opts.get('size', 18))
            f.bold = opts.get('bold', False)
            f.italic = opts.get('italic', False)
            f.color.rgb = opts.get('color', INK)
    return shape


def make_formula(out_dir, name, lines, fontsize=30):
    """Typeset one or more LaTeX (matplotlib mathtext) lines into a transparent
    PNG and return its path. Formulas as images avoid all bidi trouble."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".png")
    fig = plt.figure(figsize=(12, 0.95 * len(lines)))
    for i, ln in enumerate(lines):
        fig.text(0.5, 1 - (i + 0.5) / len(lines), f"${ln}$", ha="center", va="center",
                 fontsize=fontsize, color="#1F3A5F")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def mark_hebrew_runs(prs):
    # Without a Hebrew language tag + complex-script font, PowerPoint lays
    # out Hebrew runs as Latin text and misplaces numbers/punctuation
    # ("5,000הערכים", "-16:00?או ב"). Only Hebrew runs get the tag — on a
    # pure-English run it reverses the text. Run after bidi_fix, which
    # splits mixed runs into script-pure pieces.
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    if not _HEB_RE.search(r.text or ''):
                        continue
                    rPr = r._r.get_or_add_rPr()
                    rPr.set('lang', 'he-IL')
                    cs = rPr.find(qn('a:cs'))
                    if cs is None:
                        cs = rPr.makeelement(qn('a:cs'), {})
                        rPr.append(cs)
                    latin = rPr.find(qn('a:latin'))
                    cs.set('typeface', latin.get('typeface') if latin is not None else HEBREW_FONT)


def _fill(shape, color, line_color=None, line_w=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(line_w or 1)
    shape.shadow.inherit = False


# ----------------------------------------------------------------------------
# Deck / slide scaffolding
# ----------------------------------------------------------------------------
class Deck:
    def __init__(self, footer_label):
        self.prs = Presentation()
        self.prs.slide_width = EMU_W
        self.prs.slide_height = EMU_H
        self.blank = self.prs.slide_layouts[6]
        self.footer_label = footer_label
        self.n = 0

    def _blank_slide(self, bg=WHITE):
        s = self.prs.slides.add_slide(self.blank)
        rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
        _fill(rect, bg)
        rect.line.fill.background()
        return s

    def _box(self, s, l, t, w, h):
        return s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))

    def _panel(self, s, l, t, w, h, color=PANEL, line_color=LINE, radius=True):
        shp = s.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
            Inches(l), Inches(t), Inches(w), Inches(h))
        _fill(shp, color, line_color=line_color, line_w=0.75)
        try:
            shp.adjustments[0] = 0.06
        except Exception:
            pass
        return shp

    def _footer(self, s, dark=False):
        self.n += 1
        col = RGBColor(0xC7, 0xD3, 0xDE) if dark else MUTED
        tb = self._box(s, 0.5, 7.02, 12.33, 0.35)
        set_text(tb, [[(f"{self.footer_label}  ·  {self.n}", {'size': 10, 'color': col})]],
                 align=PP_ALIGN.LEFT, rtl=False)

    # --- slide types -------------------------------------------------------
    def title_slide(self, course, lesson_tag, title, subtitle):
        s = self._blank_slide(NAVY)
        # accent bar
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(3.05), EMU_W, Inches(0.09))
        _fill(bar, ORANGE)
        # course (small, top)
        set_text(self._box(s, 1.0, 1.1, 11.33, 0.6),
                 [[(course, {'size': 20, 'color': RGBColor(0x9F, 0xC0, 0xD8)})]],
                 align=PP_ALIGN.RIGHT)
        # lesson tag
        set_text(self._box(s, 1.0, 1.75, 11.33, 0.6),
                 [[(lesson_tag, {'size': 16, 'bold': True, 'color': ORANGE})]],
                 align=PP_ALIGN.RIGHT)
        # main title
        set_text(self._box(s, 1.0, 3.35, 11.33, 1.6),
                 [[(title, {'size': 40, 'bold': True, 'color': WHITE})]],
                 align=PP_ALIGN.RIGHT)
        # subtitle
        set_text(self._box(s, 1.0, 4.95, 11.33, 1.4),
                 [[(subtitle, {'size': 20, 'color': RGBColor(0xC7, 0xD8, 0xE6)})]],
                 align=PP_ALIGN.RIGHT, line=1.15)
        return s

    def divider(self, kicker, title):
        s = self._blank_slide(NAVY)
        band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6), Inches(0.22), Inches(2.3))
        _fill(band, ORANGE)
        set_text(self._box(s, 1.0, 2.75, 11.0, 0.7),
                 [[(kicker, {'size': 18, 'bold': True, 'color': ORANGE})]],
                 align=PP_ALIGN.RIGHT)
        set_text(self._box(s, 1.0, 3.5, 11.0, 1.6),
                 [[(title, {'size': 34, 'bold': True, 'color': WHITE})]],
                 align=PP_ALIGN.RIGHT, line=1.1)
        self._footer(s, dark=True)
        return s

    def _header(self, s, kicker, title):
        # kicker (English term, small teal)
        if kicker:
            set_text(self._box(s, 0.5, 0.45, 12.33, 0.45),
                     [[(kicker, {'size': 14, 'bold': True, 'color': TEAL})]],
                     align=PP_ALIGN.RIGHT)
        set_text(self._box(s, 0.5, 0.82, 12.33, 0.9),
                 [[(title, {'size': 28, 'bold': True, 'color': NAVY})]],
                 align=PP_ALIGN.RIGHT, line=1.0)
        rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.72),
                                  Inches(12.33), Pt(2))
        _fill(rule, TEAL)

    def content(self, title, bullets, kicker=None, lead=None):
        """bullets: list of (head, body) or plain strings."""
        s = self._blank_slide()
        self._header(s, kicker, title)
        top = 2.05
        if lead:
            tb = self._box(s, 0.5, top, 12.33, 0.9)
            set_text(tb, [[(lead, {'size': 18, 'color': MUTED, 'italic': True})]],
                     align=PP_ALIGN.RIGHT, line=1.15)
            top += 0.9
        n = len(bullets)
        gap = 0.18
        avail = 6.7 - top
        bh = (avail - gap * (n - 1)) / max(n, 1)
        bh = min(bh, 1.25)
        for i, b in enumerate(bullets):
            t = top + i * (bh + gap)
            panel = self._panel(s, 0.5, t, 12.33, bh, color=PANEL if i % 2 == 0 else PANEL2)
            tb = self._box(s, 0.9, t + 0.06, 11.5, bh - 0.12)
            if isinstance(b, (tuple, list)):
                head, body = b
                runs = [[(head + "  ", {'size': 18, 'bold': True, 'color': NAVY})]]
                if body:
                    runs.append([(body, {'size': 15, 'color': INK})])
                set_text(tb, runs, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.05)
            else:
                set_text(tb, [[(b, {'size': 17, 'color': INK})]],
                         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.05)
        self._footer(s)
        return s

    def statement(self, title, big_lines, kicker=None, sub=None, accent=TEAL):
        """A slide with one or more large centered statements (good for questions/results)."""
        if accent is None:
            accent = NAVY
        s = self._blank_slide()
        self._header(s, kicker, title)
        card = self._panel(s, 1.2, 2.4, 10.93, 3.4, color=PANEL)
        stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(12.13 - 0.0), Inches(2.4),
                                    Inches(0.0), Inches(3.4))
        runs = []
        for ln in big_lines:
            runs.append([(ln, {'size': 26, 'bold': True, 'color': accent})])
        tb = self._box(s, 1.5, 2.6, 10.33, 3.0)
        set_text(tb, runs, align=PP_ALIGN.CENTER, rtl=True, anchor=MSO_ANCHOR.MIDDLE, line=1.25)
        if sub:
            tb2 = self._box(s, 1.5, 5.95, 10.33, 0.9)
            set_text(tb2, [[(sub, {'size': 16, 'color': MUTED})]],
                     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP, line=1.1)
        self._footer(s)
        return s

    def figure(self, title, image_path, caption=None, kicker=None, note=None,
               max_w=12.0, max_h=None):
        s = self._blank_slide()
        self._header(s, kicker, title)
        top = 2.0
        bottom_reserved = 0.6 if note else 0.2
        avail_h = 6.85 - top - bottom_reserved
        if max_h is None:
            max_h = avail_h
        else:
            max_h = min(max_h, avail_h)
        w, h = Image.open(image_path).size
        ar = w / h
        draw_w = max_w
        draw_h = draw_w / ar
        if draw_h > max_h:
            draw_h = max_h
            draw_w = draw_h * ar
        left = (13.333 - draw_w) / 2
        s.shapes.add_picture(image_path, Inches(left), Inches(top),
                             Inches(draw_w), Inches(draw_h))
        y = top + draw_h + 0.1
        if caption:
            tb = self._box(s, 0.5, y, 12.33, 0.5)
            set_text(tb, [[(caption, {'size': 15, 'color': MUTED, 'italic': True})]],
                     align=PP_ALIGN.CENTER, line=1.05)
            y += 0.5
        if note:
            nb_ = self._panel(s, 1.2, min(y, 6.35), 10.93, 0.6, color=PANEL2)
            tb = self._box(s, 1.5, min(y, 6.35) + 0.03, 10.33, 0.54)
            set_text(tb, [[(note, {'size': 14, 'color': NAVY})]],
                     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.0)
        self._footer(s)
        return s

    def figure_text(self, title, image_path, bullets, kicker=None, img_h=3.0):
        """Figure on top, supporting bullets below."""
        s = self._blank_slide()
        self._header(s, kicker, title)
        top = 1.95
        w, h = Image.open(image_path).size
        ar = w / h
        draw_h = img_h
        draw_w = draw_h * ar
        if draw_w > 12.0:
            draw_w = 12.0
            draw_h = draw_w / ar
        left = (13.333 - draw_w) / 2
        s.shapes.add_picture(image_path, Inches(left), Inches(top),
                             Inches(draw_w), Inches(draw_h))
        ytop = top + draw_h + 0.2
        n = len(bullets)
        gap = 0.15
        avail = 6.8 - ytop
        bh = (avail - gap * (n - 1)) / max(n, 1)
        bh = min(bh, 1.0)
        for i, b in enumerate(bullets):
            t = ytop + i * (bh + gap)
            self._panel(s, 0.9, t, 11.53, bh, color=PANEL if i % 2 == 0 else PANEL2)
            tb = self._box(s, 1.25, t + 0.03, 10.9, bh - 0.06)
            if isinstance(b, (tuple, list)):
                head, body = b
                runs = [[(head + "  ", {'size': 16, 'bold': True, 'color': NAVY}),
                         (body, {'size': 14, 'color': INK})]]
            else:
                runs = [[(b, {'size': 15, 'color': INK})]]
            set_text(tb, runs, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.0)
        self._footer(s)
        return s

    def two_col(self, title, left_head, left_items, right_head, right_items, kicker=None):
        s = self._blank_slide()
        self._header(s, kicker, title)
        top = 2.05
        colw = 6.0
        for ci, (head, items, col) in enumerate([
                (right_head, right_items, TEAL), (left_head, left_items, ORANGE)]):
            l = 0.5 + ci * (colw + 0.33)
            self._panel(s, l, top, colw, 4.5, color=PANEL)
            bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(top),
                                     Inches(colw), Inches(0.6))
            _fill(bar, col)
            hb = self._box(s, l + 0.2, top + 0.05, colw - 0.4, 0.5)
            set_text(hb, [[(head, {'size': 18, 'bold': True, 'color': WHITE})]],
                     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
            body = self._box(s, l + 0.3, top + 0.75, colw - 0.6, 3.6)
            runs = [[("•  " + it, {'size': 16, 'color': INK})] for it in items]
            set_text(body, runs, align=PP_ALIGN.RIGHT, line=1.1, space_after=8)
        self._footer(s)
        return s

    def table(self, title, header, rows, kicker=None, note=None, highlight_last=False,
              col_w=2.2, row_h=0.7):
        """A centered data table. header/rows: lists of strings. With
        highlight_last, the last row and last column (e.g. marginals) are tinted."""
        s = self._blank_slide()
        self._header(s, kicker, title)
        n_rows, n_cols = len(rows) + 1, len(header)
        w, h = col_w * n_cols, row_h * n_rows
        left, top = (13.333 - w) / 2, 2.3
        tbl = s.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                 Inches(w), Inches(h)).table
        for r in range(n_rows):
            tbl.rows[r].height = Inches(row_h)
            for c in range(n_cols):
                cell = tbl.cell(r, c)
                text = header[c] if r == 0 else rows[r - 1][c]
                is_head = r == 0 or c == 0
                is_marg = highlight_last and r > 0 and (r == n_rows - 1 or c == n_cols - 1)
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY if r == 0 else (PANEL2 if (c == 0 or is_marg) else PANEL)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                set_text(cell, [[(text, {'size': 20, 'bold': is_head or is_marg,
                                         'color': WHITE if r == 0 else (NAVY if is_head or is_marg else INK)})]],
                         align=PP_ALIGN.CENTER, rtl=False, anchor=MSO_ANCHOR.MIDDLE)
        if note:
            y = top + h + 0.45
            self._panel(s, 1.2, y, 10.93, 0.7, color=PANEL2)
            tb = self._box(s, 1.5, y + 0.05, 10.33, 0.6)
            set_text(tb, [[(note, {'size': 15, 'color': NAVY})]],
                     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.0)
        self._footer(s)
        return s

    def notation(self, title, items, formula_dir, kicker=None):
        """Glossary slide: two columns of cards, each with a typeset symbol on the
        right and a Hebrew (head, body) explanation to its left. items are
        (name, latex, head, body), read right column first, top to bottom."""
        s = self._blank_slide()
        self._header(s, kicker, title)
        per_col = (len(items) + 1) // 2
        top, gap, colw = 2.05, 0.15, 6.0
        ch = min(1.1, (4.7 - gap * (per_col - 1)) / per_col)
        for i, (name, latex, head, body) in enumerate(items):
            col, row = divmod(i, per_col)
            l = 0.5 + (1 - col) * (colw + 0.33)
            t = top + row * (ch + gap)
            self._panel(s, l, t, colw, ch, color=PANEL if row % 2 == 0 else PANEL2)
            img = make_formula(formula_dir, "sym_" + name, [latex], fontsize=40)
            w, h = Image.open(img).size
            ih = min(0.55, ch - 0.25)
            iw = ih * w / h
            s.shapes.add_picture(img, Inches(l + colw - 0.25 - iw), Inches(t + (ch - ih) / 2),
                                 Inches(iw), Inches(ih))
            tb = self._box(s, l + 0.15, t + 0.04, colw - iw - 0.6, ch - 0.08)
            set_text(tb, [[(head, {'size': 16, 'bold': True, 'color': NAVY})],
                          [(body, {'size': 14, 'color': INK})]],
                     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, line=1.0, space_after=2)
        self._footer(s)
        return s

    def save(self, path):
        from bidi_fix import fix_presentation
        fix_presentation(self.prs)
        mark_hebrew_runs(self.prs)
        self.prs.save(path)
