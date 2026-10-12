# -*- coding: utf-8 -*-
"""
Fix Hebrew/Latin-letter mixing on a single line by inserting a line break
(soft return, <a:br/>) at each script boundary, so every visual row is
script-pure — the reliable fix for bracket/operator mirroring when English
terms or equations are embedded in Hebrew sentences.

Plain numbers embedded in Hebrew ("10,000 ממוצעים") are left inline; only
actual Latin *letters* (English words, equations, notation like GI/GI/1)
trigger a break. Safe to run in place on an already-saved .pptx — it only
rewrites paragraphs it detects as genuinely mixed, and preserves per-run
formatting (font, size, bold, color) by copying it from the original runs.
"""
import re
from copy import deepcopy
from pptx.oxml.ns import qn

_HEB_RE = re.compile(r'[֐-׿]')
_LATIN_LETTER_RE = re.compile(r'[A-Za-z]')
_CTRL_RE = re.compile(r'[‎‏‪-‮⁦-⁩]')


def _classify(tok):
    if _HEB_RE.search(tok):
        return 'HEB'
    if _LATIN_LETTER_RE.search(tok):
        return 'LTR'
    return 'WEAK'


def _insert_before_end(p_elem, el):
    end_pr = p_elem.find(qn('a:endParaRPr'))
    if end_pr is not None:
        end_pr.addprevious(el)
    else:
        p_elem.append(el)


def _split_paragraph(paragraph):
    runs = list(paragraph.runs)
    if not runs:
        return False

    pieces = []
    for run in runs:
        text = _CTRL_RE.sub('', run.text or '')
        if text == '':
            continue
        # Split at whitespace AND at script boundaries within a "word" —
        # otherwise a bound Hebrew prefix glued to a Latin word with no
        # space (e.g. "ו-Variance") is misread as one indivisible token.
        for tok in re.findall(r'[֐-׿]+|[^֐-׿\s]+|\s+', text):
            pieces.append({'text': tok, 'type': _classify(tok), 'run': run})

    # A single bound Hebrew letter (ו/ב/כ/ל/מ/ש/ה) glued directly (no
    # space) to a following English/equation token ("מ-KPI", "ל-P(Wq)")
    # is normal Hebrew usage, not mixing — fold it into that token's side
    # instead of stranding it on its own row.
    for i in range(len(pieces) - 1):
        if (pieces[i]['type'] == 'HEB' and len(pieces[i]['text']) == 1
                and pieces[i + 1]['type'] == 'LTR'):
            pieces[i]['type'] = 'LTR'

    if not pieces:
        return False
    has_heb = any(p['type'] == 'HEB' for p in pieces)
    has_ltr = any(p['type'] == 'LTR' for p in pieces)
    if not (has_heb and has_ltr):
        return False  # not genuinely mixed (plain numbers don't count)

    n = len(pieces)
    side = [None] * n
    last_strong = None
    for i, p in enumerate(pieces):
        if p['type'] in ('HEB', 'LTR'):
            side[i] = p['type']
            last_strong = p['type']
        else:
            side[i] = last_strong
    next_strong = None
    for i in range(n - 1, -1, -1):
        if pieces[i]['type'] in ('HEB', 'LTR'):
            next_strong = pieces[i]['type']
        elif side[i] is None:
            side[i] = next_strong

    groups = []
    for i, p in enumerate(pieces):
        if groups and groups[-1][0] == side[i]:
            groups[-1][1].append(p)
        else:
            groups.append([side[i], [p]])

    cleaned = []
    for s, gp in groups:
        while gp and gp[0]['type'] == 'WEAK' and gp[0]['text'].strip() == '':
            gp = gp[1:]
        while gp and gp[-1]['type'] == 'WEAK' and gp[-1]['text'].strip() == '':
            gp = gp[:-1]
        if gp:
            cleaned.append((s, gp))
    if len(cleaned) < 2:
        return False

    p_elem = paragraph._p
    for r in p_elem.findall(qn('a:r')):
        p_elem.remove(r)

    for gi, (side_, gp) in enumerate(cleaned):
        cur_run = None
        cur_text = []

        def flush():
            if cur_run is not None:
                r_el = deepcopy(cur_run._r)
                t_el = r_el.find(qn('a:t'))
                if t_el is None:
                    t_el = r_el.makeelement(qn('a:t'), {})
                    r_el.append(t_el)
                t_el.text = ''.join(cur_text)
                _insert_before_end(p_elem, r_el)

        for piece in gp:
            if cur_run is piece['run']:
                cur_text.append(piece['text'])
            else:
                flush()
                cur_run = piece['run']
                cur_text = [piece['text']]
        flush()
        if gi < len(cleaned) - 1:
            br_el = p_elem.makeelement(qn('a:br'), {})
            _insert_before_end(p_elem, br_el)

    return True


def fix_presentation(prs):
    """Split every mixed Hebrew/Latin-letter paragraph into script-pure
    rows. Returns the number of paragraphs changed."""
    changed = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                if _split_paragraph(paragraph):
                    changed += 1
    return changed
