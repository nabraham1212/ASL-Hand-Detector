"""Rule-based ASL coach for the 24 supported static letters.

This is beginner guidance, not official ASL instruction. The tips emphasize the
features that separate commonly-confused letters (thumb position, fingers
together/apart/crossed, open vs. closed). It reacts to the model's confidence
and stability; it does NOT analyze your exact hand geometry.

Pure logic only: no OpenCV, no colors. The UI layer decides how to display this.
"""

SUPPORTED_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # 24 letters, no J/Z

# Per-letter hints for all 24 supported letters. Hard/confusable letters get
# extra detail about the distinguishing feature.
LETTER_TIPS = {
    "A": "Closed fist with your thumb on the SIDE of your fingers, not across the front.",
    "B": "Four fingers straight up and together; thumb folded across your palm.",
    "C": "Curve your whole hand into a C - open and rounded, not closed.",
    "D": "Index finger straight up; other fingertips meet your thumb in a circle.",
    "E": "Curl all four fingers down to the palm with the thumb tucked against them.",
    "F": "Thumb and index fingertip touch in a circle; the other three fingers point up.",
    "G": "Hand sideways; index finger and thumb point flat to the side with a small gap.",
    "H": "Hand sideways; index and middle fingers TOGETHER, pointing to the side.",
    "I": "Only your pinky points up; the rest of the fingers make a fist.",
    "K": "Index and middle fingers up in a V, with the thumb resting BETWEEN them.",
    "L": "Thumb out and index finger up to form an L; the other fingers fold down.",
    "M": "Thumb tucked UNDER THREE fingers (index, middle, ring fold over it).",
    "N": "Thumb tucked UNDER TWO fingers (index and middle fold over it).",
    "O": "All fingertips curve to meet the thumb, forming a closed O / circle.",
    "P": "Like K but pointed DOWN: index and middle out, thumb between them.",
    "Q": "Like G but pointed DOWN: thumb and index point down with a small gap.",
    "R": "Index and middle fingers CROSSED over each other, pointing up.",
    "S": "Closed fist with your thumb wrapped ACROSS THE FRONT of your fingers.",
    "T": "Thumb tucked BETWEEN your index and middle fingers.",
    "U": "Index and middle fingers up and TOGETHER (touching), pointing up.",
    "V": "Index and middle fingers up and APART, like a peace sign.",
    "W": "Three fingers up and spread apart: index, middle, and ring.",
    "X": "Index finger up and bent into a hook; the other fingers fold down.",
    "Y": "Thumb and pinky stretched out, the middle three fingers folded (hang loose).",
}

# Letters that are commonly confused and deserve extra attention on screen.
HARD_LETTERS = {"A", "E", "M", "N", "S", "T", "G", "H", "U", "V", "R", "K", "O", "C"}


def get_letter_tip(letter):
    """Per-letter hint, or '' if the letter isn't a supported static letter."""
    if not letter:
        return ""
    return LETTER_TIPS.get(letter.upper(), "")


def is_hard_letter(letter):
    """True if the letter is in a commonly-confused group."""
    return bool(letter) and letter.upper() in HARD_LETTERS


def get_general_tip(confidence):
    """General improvement tip when the sign is unclear / low confidence."""
    if confidence is None:
        return "Make a clear letter sign in the frame."
    if confidence < 0.40:
        return "Move your hand fully into frame and improve your lighting."
    return "Hold your hand steady and keep it away from your face/body."


def get_coach_feedback(raw_prediction, smoothed_prediction, confidence,
                       hand_detected, is_stable, candidate_letter, threshold):
    """Return one short situational coach line.

    Priority:
      1. No hand          -> show a hand
      2. Stable & waiting  -> confirm or retry
      3. No confidence yet -> make a clear sign
      4. Low confidence    -> a general improvement tip
      5. Jittering         -> hold still
      6. Otherwise         -> good, hold steady
    """
    if not hand_detected:
        return "Show one hand clearly in the frame."
    if is_stable and candidate_letter and candidate_letter != "-":
        return f"Stable on {candidate_letter}. Press Y to confirm or N to retry."
    if confidence is None:
        return "Make a clear letter sign."
    if confidence < threshold:
        return get_general_tip(confidence)
    if raw_prediction != smoothed_prediction:
        return "Almost there - hold the sign still."
    return "Good shape. Hold steady to confirm."