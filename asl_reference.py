"""In-game ASL Reference Guide (text-only, no images).

Pages through all 24 supported static letters. Letter descriptions are pulled
from asl_coach.LETTER_TIPS so there is a single source of truth and the guide
can never disagree with the coach. This module adds short "common confusion"
notes and a final page summarizing the tricky letter groups.

J and Z are intentionally absent (they need motion - coming later).
"""

import cv2
from asl_coach import LETTER_TIPS, SUPPORTED_LETTERS

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)

LETTERS_PER_PAGE = 8

# Short "watch out / vs" notes for the easily-confused letters only.
# Letters without an entry just show their description with no extra line.
CONFUSION_NOTES = {
    "A": "vs S/T/E - thumb on the SIDE.",
    "C": "vs O - OPEN, with a gap.",
    "D": "vs F - index up, other tips meet thumb.",
    "E": "vs A/S - fingers curl down, thumb at palm.",
    "F": "vs D - thumb+index circle, three fingers up.",
    "G": "vs H - INDEX only, hand sideways.",
    "H": "vs G - index AND middle, hand sideways.",
    "K": "vs V - V shape with thumb BETWEEN fingers.",
    "M": "vs N - thumb under THREE fingers.",
    "N": "vs M/T - thumb under TWO fingers.",
    "O": "vs C - CLOSED circle.",
    "P": "vs K - same shape pointing DOWN.",
    "Q": "vs G - same shape pointing DOWN.",
    "R": "vs U/V - fingers CROSSED.",
    "S": "vs A/T - thumb ACROSS the front.",
    "T": "vs A/S/N - thumb BETWEEN index & middle.",
    "U": "vs V/R - fingers TOGETHER.",
    "V": "vs U/R - fingers APART.",
    "X": "index finger bent into a hook.",
}

# Summary notes for the confusing groups (shown on the final page).
GROUP_NOTES = [
    ("A E M N S T  (fist family)",
     "Thumb position is everything: A=side, S=front, T=between, "
     "M=under 3 fingers, N=under 2, E=curled in."),
    ("G H  (sideways hand)",
     "G = index finger only. H = index AND middle together."),
    ("U V R K  (two-finger shapes)",
     "U=together, V=apart, R=crossed, K=V-shape with the thumb between."),
    ("C O  (rounded hand)",
     "C is open with a visible gap. O is closed - fingertips touch the thumb."),
]


def _letter_pages():
    return [SUPPORTED_LETTERS[i:i + LETTERS_PER_PAGE]
            for i in range(0, len(SUPPORTED_LETTERS), LETTERS_PER_PAGE)]


def get_total_pages():
    """Letter pages + 1 'tricky groups' page."""
    return len(_letter_pages()) + 1


def _wrap(text, max_chars):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_reference_overlay(frame, page):
    """Draw the semi-transparent reference guide for the given 0-indexed page."""
    h, w = frame.shape[:2]
    total = get_total_pages()
    page = max(0, min(page, total - 1))
    max_chars = max(40, (w - 90) // 10)

    # Dark translucent backdrop over the webcam.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title.
    cv2.putText(frame, "ASL Reference Guide", (40, 46), FONT, 1.0, MAGENTA, 2, cv2.LINE_AA)
    cv2.putText(frame, "J and Z are motion letters - coming later",
                (40, 74), FONT, 0.55, GRAY, 1, cv2.LINE_AA)

    letter_pages = _letter_pages()
    y = 108

    if page < len(letter_pages):
        for L in letter_pages[page]:
            tip = LETTER_TIPS.get(L, "")
            cv2.putText(frame, f"{L}:  {tip}", (40, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
            y += 25
            note = CONFUSION_NOTES.get(L, "")
            if note:
                cv2.putText(frame, f"      watch: {note}", (40, y), FONT, 0.45, GRAY, 1, cv2.LINE_AA)
                y += 23
            y += 6
    else:
        # Final page: confusing-group summary.
        cv2.putText(frame, "Tricky Groups - focus on these features",
                    (40, y), FONT, 0.6, YELLOW, 1, cv2.LINE_AA)
        y += 36
        for title, advice in GROUP_NOTES:
            cv2.putText(frame, title, (40, y), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
            y += 26
            for ln in _wrap(advice, max_chars):
                cv2.putText(frame, "   " + ln, (40, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
                y += 24
            y += 8

    # Footer: page number + controls.
    cv2.putText(frame, f"Page {page + 1}/{total}", (40, h - 50), FONT, 0.6, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, "[ prev page    ] next page    G close guide",
                (40, h - 22), FONT, 0.55, GRAY, 1, cv2.LINE_AA)