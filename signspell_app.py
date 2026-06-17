"""
SignSpell - ASL Hangman Learning Game
=====================================
Play Hangman by finger-spelling letters in ASL. The app recognizes the letter,
asks you to confirm it, and submits it as a guess. Each round is a cybersecurity
or technology term, so you learn vocabulary while you practice fingerspelling.

Supported: 24 static ASL letters (A-Y). J and Z are NOT supported yet because
they require motion.

Run:  .\\.venv312\\Scripts\\python.exe signspell_app.py
"""

import os
import sys
import json
import time
import random
import warnings
from collections import deque, Counter

import cv2
import numpy as np
import joblib

import HandTrackingModule as htm
from hangman_game import HangmanGame
from asl_utils import extract_normalized_landmarks, EXPECTED_FEATURES
from asl_coach import get_coach_feedback, get_letter_tip, is_hard_letter
from asl_reference import draw_reference_overlay, get_total_pages

# Fixed feature order matches training, so this benign sklearn warning is silenced.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---- Config ----
MODEL_PATH = os.path.join("models", "asl_model.pkl")
INFO_PATH = os.path.join("models", "asl_model_info.json")

CONFIDENCE_THRESHOLD = 0.70   # 24 classes -> lower top probability than a 9-letter model
HOLD_SECONDS = 1.0            # how long a sign must stay stable before we ask to confirm
CONFIRM_COOLDOWN = 1.0        # pause after a confirm so you don't instantly re-confirm
SMOOTHING_WINDOW = 10
THRESHOLD_STEP = 0.05

# The 24 supported static letters (no J, no Z).
SUPPORTED_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")

# ---- Colors (BGR) ----
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
CYAN = (255, 255, 0)


# =====================================================================
# Term bank: each round is a cyber/tech word you also learn the meaning of.
#   answer     -> uppercase Hangman answer (letters only, no spaces/hyphens, no J/Z)
#   display    -> friendly name (may contain spaces)
#   category   -> Cybersecurity / Programming / Networking / Technology
#   definition -> short beginner-friendly meaning (clue; avoids the answer word)
# =====================================================================
TERM_BANK = [
    # --- Cybersecurity ---
    {"answer": "FIREWALL", "display": "Firewall", "category": "Cybersecurity",
     "definition": "A security system that monitors and filters network traffic to block unwanted access."},
    {"answer": "MALWARE", "display": "Malware", "category": "Cybersecurity",
     "definition": "Harmful software designed to damage, disrupt, or gain unauthorized access to a system."},
    {"answer": "PHISHING", "display": "Phishing", "category": "Cybersecurity",
     "definition": "A scam that tricks people into giving up sensitive info by pretending to be trustworthy."},
    {"answer": "PASSWORD", "display": "Password", "category": "Cybersecurity",
     "definition": "A secret string of characters used to verify who you are and protect an account."},
    {"answer": "ENCRYPTION", "display": "Encryption", "category": "Cybersecurity",
     "definition": "Scrambling data so only someone with the right key can read it."},
    {"answer": "HASHING", "display": "Hashing", "category": "Cybersecurity",
     "definition": "Turning data into a fixed-length fingerprint that cannot be easily reversed."},
    {"answer": "BACKUP", "display": "Backup", "category": "Cybersecurity",
     "definition": "A saved copy of data you can restore if the original is lost."},
    {"answer": "PATCH", "display": "Patch", "category": "Cybersecurity",
     "definition": "A software update that fixes bugs or closes security holes."},
    {"answer": "EXPLOIT", "display": "Exploit", "category": "Cybersecurity",
     "definition": "Code or a technique that takes advantage of a weakness in a system."},
    {"answer": "VIRUS", "display": "Virus", "category": "Cybersecurity",
     "definition": "Malicious code that spreads by attaching itself to other files or programs."},
    {"answer": "THREAT", "display": "Threat", "category": "Cybersecurity",
     "definition": "Anything that could potentially cause harm to a system or its data."},
    {"answer": "BREACH", "display": "Breach", "category": "Cybersecurity",
     "definition": "An incident where protected data is accessed or stolen without permission."},
    {"answer": "FORENSICS", "display": "Forensics", "category": "Cybersecurity",
     "definition": "Investigating digital evidence after an incident to find out what happened."},
    {"answer": "LOGIN", "display": "Login", "category": "Cybersecurity",
     "definition": "The process of signing in to prove your identity to a system."},
    {"answer": "TOKEN", "display": "Token", "category": "Cybersecurity",
     "definition": "A small piece of data used to prove identity or grant temporary access."},
    {"answer": "COOKIE", "display": "Cookie", "category": "Cybersecurity",
     "definition": "A small file a website stores in your browser to remember information about you."},
    {"answer": "SANDBOX", "display": "Sandbox", "category": "Cybersecurity",
     "definition": "An isolated space where untrusted code can run safely without affecting the system."},
    {"answer": "BOTNET", "display": "Botnet", "category": "Cybersecurity",
     "definition": "A network of infected computers controlled remotely by an attacker."},
    {"answer": "SPYWARE", "display": "Spyware", "category": "Cybersecurity",
     "definition": "Software that secretly gathers information about a user without their knowledge."},
    {"answer": "ADWARE", "display": "Adware", "category": "Cybersecurity",
     "definition": "Software that automatically shows or downloads unwanted advertisements."},
    {"answer": "PHARMING", "display": "Pharming", "category": "Cybersecurity",
     "definition": "An attack that redirects users to fake websites to steal their information."},
    {"answer": "PENTEST", "display": "Penetration Test", "category": "Cybersecurity",
     "definition": "An authorized simulated attack used to find weaknesses before real attackers do."},
    {"answer": "HACKER", "display": "Hacker", "category": "Cybersecurity",
     "definition": "A person who uses technical skills to access or manipulate systems."},
    {"answer": "SPOOFING", "display": "Spoofing", "category": "Cybersecurity",
     "definition": "Pretending to be a trusted source by faking an address or identity."},
    {"answer": "SNIFFING", "display": "Sniffing", "category": "Cybersecurity",
     "definition": "Secretly capturing data as it travels across a network."},
    {"answer": "SCANNING", "display": "Scanning", "category": "Cybersecurity",
     "definition": "Probing a system or network to discover open ports and weaknesses."},
    {"answer": "CIPHER", "display": "Cipher", "category": "Cybersecurity",
     "definition": "A method for transforming readable text into a secret coded form."},
    {"answer": "SECURITY", "display": "Security", "category": "Cybersecurity",
     "definition": "The practice of protecting systems and data from harm or unauthorized access."},
    {"answer": "PRIVACY", "display": "Privacy", "category": "Cybersecurity",
     "definition": "The right to control how your personal information is collected and used."},
    {"answer": "CREDENTIALS", "display": "Credentials", "category": "Cybersecurity",
     "definition": "The information, like a username and password, used to prove identity."},
    {"answer": "ACCESSCONTROL", "display": "Access Control", "category": "Cybersecurity",
     "definition": "Rules that decide who is allowed to use or view a resource."},
    {"answer": "BRUTEFORCE", "display": "Brute Force", "category": "Cybersecurity",
     "definition": "Trying many combinations rapidly to guess a password or key."},
    {"answer": "PACKET", "display": "Packet", "category": "Networking",
     "definition": "A small unit of data sent across a network."},
    {"answer": "ROUTER", "display": "Router", "category": "Networking",
     "definition": "A device that directs data between different networks."},
    {"answer": "PROTOCOL", "display": "Protocol", "category": "Networking",
     "definition": "A set of rules that lets computers communicate over a network."},
    # --- Programming ---
    {"answer": "PYTHON", "display": "Python", "category": "Programming",
     "definition": "A popular, beginner-friendly programming language used widely in AI and security."},
    {"answer": "PROGRAM", "display": "Program", "category": "Programming",
     "definition": "A set of instructions that tells a computer how to perform a task."},
    {"answer": "VARIABLE", "display": "Variable", "category": "Programming",
     "definition": "A named container that stores a value a program can use and change."},
    {"answer": "FUNCTION", "display": "Function", "category": "Programming",
     "definition": "A reusable block of code that performs a specific task."},
    {"answer": "COMPILER", "display": "Compiler", "category": "Programming",
     "definition": "A tool that translates source code into a form the computer can run."},
    {"answer": "BINARY", "display": "Binary", "category": "Programming",
     "definition": "The base-2 number system of 0s and 1s that computers use."},
    {"answer": "DATABASE", "display": "Database", "category": "Programming",
     "definition": "An organized collection of data that can be easily stored and retrieved."},
    {"answer": "TERMINAL", "display": "Terminal", "category": "Programming",
     "definition": "A text interface where you type commands to control a computer."},
    # --- Networking ---
    {"answer": "NETWORK", "display": "Network", "category": "Networking",
     "definition": "A group of connected computers that can share data and resources."},
    {"answer": "SERVER", "display": "Server", "category": "Networking",
     "definition": "A computer that provides data or services to other computers."},
    {"answer": "CLIENT", "display": "Client", "category": "Networking",
     "definition": "A computer or app that requests data or services from a server."},
    # --- Technology ---
    {"answer": "KEYBOARD", "display": "Keyboard", "category": "Technology",
     "definition": "An input device with keys used to type text and commands."},
    {"answer": "MONITOR", "display": "Monitor", "category": "Technology",
     "definition": "A screen that displays a computer's visual output."},
    {"answer": "COMPUTER", "display": "Computer", "category": "Technology",
     "definition": "An electronic device that processes data and runs programs."},
    {"answer": "HARDWARE", "display": "Hardware", "category": "Technology",
     "definition": "The physical parts of a computer you can touch."},
    {"answer": "SOFTWARE", "display": "Software", "category": "Technology",
     "definition": "Programs and instructions that tell hardware what to do."},
    {"answer": "BROWSER", "display": "Browser", "category": "Technology",
     "definition": "An application used to access and view websites."},
    {"answer": "CAMERA", "display": "Camera", "category": "Technology",
     "definition": "A device that captures images or video; this game uses it for hand tracking."},
    {"answer": "ROBOT", "display": "Robot", "category": "Technology",
     "definition": "A machine that can carry out tasks automatically, often programmable."},
    {"answer": "LAPTOP", "display": "Laptop", "category": "Technology",
     "definition": "A portable computer combining screen, keyboard, and battery in one unit."},
    {"answer": "GESTURE", "display": "Gesture", "category": "Technology",
     "definition": "A hand or body movement used as input, like the signs in this game."},
]


def validate_term_bank(bank):
    """Keep only terms whose answer is letters-only and uses supported letters
    (no spaces/hyphens, no J/Z). Returns (valid_terms, removed_messages)."""
    valid, removed = [], []
    for t in bank:
        ans = t["answer"].upper()
        if ans.isalpha() and set(ans) <= SUPPORTED_LETTERS:
            valid.append(t)
        else:
            bad = sorted(set(ans) - SUPPORTED_LETTERS) or ["non-letter"]
            removed.append(f"{t['answer']} (unsupported: {','.join(bad)})")
    return valid, removed


def pick_term(terms):
    return random.choice(terms)


# =====================================================================
# Confirmation state machine (same logic as the phase files).
# =====================================================================
class ConfirmationFSM:
    def __init__(self, hold_seconds=HOLD_SECONDS, cooldown=CONFIRM_COOLDOWN):
        self.state = "TRACKING"
        self.candidate = None
        self.hold_start = None
        self.pending = None
        self.rejected_letter = None
        self.cooldown_until = 0.0
        self.hold_seconds = hold_seconds
        self.cooldown = cooldown

    def update(self, now, hand_detected, confidence, smoothed, threshold):
        if self.state == "WAITING_FOR_CONFIRMATION":
            return  # frozen until the user answers Y/N
        if not hand_detected:
            self._reset_hold("NO_HAND")
        elif confidence is None or confidence < threshold:
            self._reset_hold("LOW_CONFIDENCE")
        elif now < self.cooldown_until:
            self._reset_hold("TRACKING")
        elif smoothed is None:
            self._reset_hold("TRACKING")
        elif self.rejected_letter is not None and smoothed == self.rejected_letter:
            self._reset_hold("TRACKING")
        else:
            self.rejected_letter = None
            if self.candidate != smoothed:
                self.candidate = smoothed
                self.hold_start = now
                self.state = "HOLDING_CANDIDATE"
            elif now - self.hold_start >= self.hold_seconds:
                self.pending = self.candidate
                self.state = "WAITING_FOR_CONFIRMATION"
            else:
                self.state = "HOLDING_CANDIDATE"

    def _reset_hold(self, new_state):
        self.state = new_state
        self.candidate = None
        self.hold_start = None

    def confirm(self, now):
        if self.state == "WAITING_FOR_CONFIRMATION" and self.pending:
            letter = self.pending
            self.cooldown_until = now + self.cooldown
            self.candidate = None
            self.hold_start = None
            self.pending = None
            self.state = "TRACKING"
            return letter
        return None

    def reject(self, now):
        if self.state == "WAITING_FOR_CONFIRMATION" and self.pending:
            letter = self.pending
            self.rejected_letter = letter
            self.candidate = None
            self.hold_start = None
            self.pending = None
            self.state = "TRACKING"
            return letter
        return None

    def clear(self):
        self.candidate = None
        self.hold_start = None
        self.pending = None
        self.rejected_letter = None
        self.state = "TRACKING"

    def progress(self, now):
        if self.state == "HOLDING_CANDIDATE" and self.hold_start is not None:
            return min(1.0, (now - self.hold_start) / self.hold_seconds)
        if self.state == "WAITING_FOR_CONFIRMATION":
            return 1.0
        return 0.0


# =====================================================================
# Model + prediction
# =====================================================================
def load_model():
    """Load model (hard error if missing) and metadata (soft warning). Exits with
    a clear message on missing model or feature-count mismatch."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run train_asl_model.py first to create it.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    info = None
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            info = json.load(f)
    else:
        print(f"Warning: metadata not found at {INFO_PATH}. Continuing without it.")

    model_expected = int(getattr(model, "n_features_in_", EXPECTED_FEATURES))
    if model_expected != EXPECTED_FEATURES:
        print(f"ERROR: feature mismatch - model expects {model_expected} features, "
              f"but the live extractor returns {EXPECTED_FEATURES}.")
        print("Retrain with train_asl_model.py so the model matches asl_utils.py.")
        sys.exit(1)

    return model, info


def predict_letter(model, features):
    proba = model.predict_proba([features])[0]
    idx = int(np.argmax(proba))
    return str(model.classes_[idx]), float(proba[idx])


def submit_confirmed_letter(game, letter):
    """Send a confirmed letter into Hangman. Returns a short status message.
    A wrong guess still costs a life, but the wording makes clear the SIGN was
    accepted - the letter simply isn't in the word."""
    if game.is_won() or game.is_lost():
        return "Game over - press R for a new word"
    result = game.guess_letter(letter)
    if result["repeated"]:
        return f"Already guessed {result['letter']}."
    if not result["valid"]:
        return "Invalid letter."
    if result["correct"]:
        return f"Correct! {result['letter']} is in the word."
    return "That letter is not in the word. Your sign was accepted."


# =====================================================================
# UI helpers
# =====================================================================
def fmt(items):
    return ", ".join(items) if items else "-"


def _wrap_to_width(text, max_px, font_scale, thickness=1):
    """Wrap one string so each line fits within max_px pixels (measured, not guessed)."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if tw <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _wrap_lines(lines, width, font_scale, pad=12):
    """Expand a list of (text, color) into wrapped (text, color) sub-lines that
    fit inside a panel of the given width. Blank lines are preserved as spacers."""
    max_px = width - pad * 2
    out = []
    for text, color in lines:
        if text == "":
            out.append(("", color))
        else:
            for sub in _wrap_to_width(text, max_px, font_scale):
                out.append((sub, color))
    return out


def draw_text_panel(img, lines, x, y, width, font_scale=0.55, line_height=26, pad=12):
    # Wrap every line to the panel width first, then size the box to fit.
    wrapped = _wrap_lines(lines, width, font_scale, pad)
    height = pad * 2 + line_height * len(wrapped)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x - pad + width, y - pad + height),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (text, color) in enumerate(wrapped):
        ty = y + line_height * i + 17
        cv2.putText(img, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
    return y - pad + height


def draw_progress_bar(img, x, y, w, h, progress):
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 2)
    fill = int(w * progress)
    if fill > 0:
        color = GREEN if progress >= 1.0 else YELLOW
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, cv2.FILLED)


def draw_word_panel(img, game, message, x=10, y=10, width=320):
    """Top-left: Hangman state (no branding header, for a cleaner UI)."""
    s = game.get_status()
    lives = s["lives_remaining"]
    lines = [
        (f"Word: {s['masked_word']}", WHITE),
        (f"Wrong: {fmt(s['wrong_guesses'])}", RED),
        (f"Guessed: {fmt(s['guessed_letters'][-12:])}", GRAY),
        (f"Lives: {lives}/{s['max_lives']}", GREEN if lives > 2 else RED),
        ("", WHITE),
    ]
    if s["won"]:
        lines.append(("Game: YOU WON!", GREEN))
        lines.append(("Press R for a new word", GREEN))
    elif s["lost"]:
        lines.append(("Game: YOU LOST", RED))
        lines.append(("Press R for a new word", RED))
    else:
        lines.append(("Game: Playing", GREEN))
        lines.append((f"Msg: {message}", YELLOW))
    return draw_text_panel(img, lines, x, y, width)


def draw_meaning_panel(img, term, game_over, x, y, width=320):
    """Word Meaning: category + definition (clue). Reveals the term on win/loss.
    The definition is passed as a single line; draw_text_panel wraps it to width."""
    lines = [("WORD MEANING", CYAN)]
    if game_over:
        lines.append((f"Answer: {term['display']}", GREEN))
    lines.append((f"Category: {term['category']}", YELLOW))
    lines.append((term["definition"], WHITE))
    draw_text_panel(img, lines, x, y, width, font_scale=0.5, line_height=22)


def draw_asl_ui(img, frame_w, hand_detected, raw_pred, confidence, smoothed,
                candidate_disp, progress, status, status_color, threshold,
                draw_landmarks, width=320):
    x = frame_w - width - 10
    y = 10
    conf_text = f"{confidence * 100:.0f}%" if confidence is not None else "-"
    conf_color = GRAY if confidence is None else (GREEN if confidence >= threshold else RED)

    # Split a "head - hint" status into two clean lines so it never overflows.
    if " - " in status:
        head, hint = status.split(" - ", 1)
        status_lines = [(f"Status: {head}", status_color), (f"Hint: {hint}", status_color)]
    else:
        status_lines = [(f"Status: {status}", status_color)]

    lines = [
        ("ASL DETECTION", MAGENTA),
        ("", WHITE),
        (f"Hand Detected: {'Yes' if hand_detected else 'No'}", GREEN if hand_detected else RED),
        (f"Raw Prediction: {raw_pred}", WHITE),
        (f"Confidence: {conf_text}", conf_color),
        (f"Smoothed: {smoothed or '-'}", WHITE),
        (f"Candidate: {candidate_disp}", YELLOW),
        (f"Hold Progress: {int(progress * 100)}%", WHITE),
    ]
    lines.extend(status_lines)
    lines.append(("", WHITE))
    lines.append((f"Threshold: {threshold:.2f}   Draw: {'ON' if draw_landmarks else 'OFF'}", GRAY))

    bottom = draw_text_panel(img, lines, x, y, width)
    draw_progress_bar(img, x, bottom + 10, width, 18, progress)


def draw_controls(img, frame_h):
    lines = [
        ("Y confirm | N reject | C clear | D draw | T tips", GRAY),
        ("+/- threshold | R new word | G guide | Q quit", GRAY),
    ]
    draw_text_panel(img, lines, 10, frame_h - 78, 520, font_scale=0.5, line_height=24)


def draw_coach_ui(img, frame_w, frame_h, coach_msg, coach_color, tip_letter,
                  tip_text, rejected_letter, rejected_tip):
    width = frame_w - 20
    lines = [("ASL COACH   (press T to hide)", MAGENTA),
             (coach_msg, coach_color)]
    if tip_letter and tip_text:
        prefix = "Tricky " if is_hard_letter(tip_letter) else "Tip "
        lines.append((f"{prefix}[{tip_letter}]: {tip_text}", YELLOW))
    if rejected_letter and rejected_tip:
        lines.append((f"Adjust [{rejected_letter}]: {rejected_tip}", RED))
    lines.append(("", WHITE))
    lines.append(("Y confirm  N reject  C clear  D draw  +/- thr  R new  T tips  G guide  Q quit", GRAY))

    # Size the panel to its real wrapped height so it sits flush at the bottom.
    line_height, pad = 24, 12
    wrapped = _wrap_lines(lines, width, font_scale=0.5, pad=pad)
    panel_height = pad * 2 + line_height * len(wrapped)
    y = frame_h - panel_height - 4
    draw_text_panel(img, lines, 10, y, width, font_scale=0.5, line_height=line_height, pad=pad)


def asl_status(fsm, game_over):
    if game_over:
        return "Game over - press R", GRAY
    if fsm.state == "NO_HAND":
        return "No hand detected", RED
    if fsm.state == "LOW_CONFIDENCE":
        return "Low confidence - adjust hand", RED
    if fsm.state == "HOLDING_CANDIDATE":
        return "Hold steady...", YELLOW
    if fsm.state == "WAITING_FOR_CONFIRMATION":
        return f"Confirm {fsm.pending}? Press Y/N", GREEN
    return "Tracking...", YELLOW


# =====================================================================
# Main
# =====================================================================
def main():
    model, info = load_model()
    acc = info.get("test_accuracy") if info else None

    terms, removed = validate_term_bank(TERM_BANK)
    if removed:
        print(f"Removed {len(removed)} invalid term(s): {removed}")
    if not terms:
        print("ERROR: no valid terms in the term bank. Add supported-letter words.")
        sys.exit(1)

    print("=== SignSpell - ASL Hangman Learning Game ===")
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Known letters ({len(model.classes_)}): {list(model.classes_)}")
    if acc is not None:
        print(f"Model accuracy (from training): {acc}")
    print("Supported: 24 static letters. J and Z need motion - not supported yet.")
    print(f"Term bank: {len(terms)} valid terms.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open the webcam.")
        print("Close other apps using the camera, check permissions, and try again.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)   # requested; falls back if unsupported
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = htm.handDetector(maxHands=1)

    current_term = pick_term(terms)
    game = HangmanGame(word=current_term["answer"])
    fsm = ConfirmationFSM()
    history = deque(maxlen=SMOOTHING_WINDOW)
    threshold = CONFIDENCE_THRESHOLD
    draw_landmarks = True
    coach_on = True
    guide_open = False
    guide_page = 0
    last_rejected = None
    last_rejected_until = 0.0
    message = "Make a sign and press Y to guess."

    print(f"New word started ({current_term['category']}, {len(game.word)} letters).")

    while True:
        now = time.time()
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)   # mirror to match training data
        frame_h, frame_w = img.shape[:2]

        img = detector.findHands(img, draw=draw_landmarks)
        raw = detector.get_raw_landmarks(0)
        hand_detected = raw is not None

        raw_pred = "-"
        confidence = None
        if hand_detected:
            features = extract_normalized_landmarks(raw)
            if features is not None and len(features) == EXPECTED_FEATURES:
                raw_pred, confidence = predict_letter(model, features)
                history.append(raw_pred)

        smoothed = Counter(history).most_common(1)[0][0] if history else None

        game_over = game.is_won() or game.is_lost()
        # Pause the confirmation engine when the game is over or the guide is open.
        if not game_over and not guide_open:
            fsm.update(now, hand_detected, confidence, smoothed, threshold)
        progress = fsm.progress(now) if not game_over else 0.0

        candidate_disp = fsm.pending if fsm.state == "WAITING_FOR_CONFIRMATION" else (fsm.candidate or "-")
        status, status_color = asl_status(fsm, game_over)
        is_stable = (fsm.state == "WAITING_FOR_CONFIRMATION")

        # --- coach feedback ---
        coach_msg = get_coach_feedback(raw_pred, smoothed, confidence,
                                       hand_detected, is_stable, candidate_disp, threshold)
        if not hand_detected:
            coach_clr = RED
        elif is_stable:
            coach_clr = GREEN
        elif confidence is None or confidence < threshold:
            coach_clr = RED
        else:
            coach_clr = YELLOW
        if fsm.state == "WAITING_FOR_CONFIRMATION":
            tip_letter = fsm.pending
        elif fsm.candidate:
            tip_letter = fsm.candidate
        elif hand_detected and smoothed:
            tip_letter = smoothed
        else:
            tip_letter = None
        tip_text = get_letter_tip(tip_letter) if tip_letter else ""
        if last_rejected and now < last_rejected_until:
            rej_letter, rej_tip = last_rejected, get_letter_tip(last_rejected)
        else:
            rej_letter, rej_tip = None, ""

        # --- draw panels ---
        word_bottom = draw_word_panel(img, game, message)
        draw_meaning_panel(img, current_term, game_over, x=10, y=word_bottom + 10)
        draw_asl_ui(img, frame_w, hand_detected, raw_pred, confidence, smoothed,
                    candidate_disp, progress, status, status_color, threshold, draw_landmarks)
        if coach_on:
            draw_coach_ui(img, frame_w, frame_h, coach_msg, coach_clr,
                          tip_letter, tip_text, rej_letter, rej_tip)
        else:
            draw_controls(img, frame_h)

        # Reference guide overlays everything when open.
        if guide_open:
            draw_reference_overlay(img, guide_page)

        cv2.imshow("SignSpell - ASL Hangman Learning Game", img)

        # --- keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            guide_open = not guide_open
        elif key == ord("d"):
            draw_landmarks = not draw_landmarks
        elif guide_open:
            # While the guide is open, only page navigation works (gameplay paused).
            if key == ord("["):
                guide_page = max(0, guide_page - 1)
            elif key == ord("]"):
                guide_page = min(get_total_pages() - 1, guide_page + 1)
        elif key == ord("t"):
            coach_on = not coach_on
        elif key == ord("c"):
            history.clear()
            fsm.clear()
            message = "Cleared prediction."
        elif key == ord("r"):
            current_term = pick_term(terms)
            game.reset(word=current_term["answer"])
            history.clear()
            fsm.clear()
            message = "New word! Make a sign."
            print(f"New word started ({current_term['category']}, {len(game.word)} letters).")
        elif key == ord("y"):
            letter = fsm.confirm(now)
            if letter:
                history.clear()
                print(f"[CONFIRMED] {letter}")
                message = submit_confirmed_letter(game, letter)
                print(f"[GUESS] {letter} -> {message}")
                if game.is_won():
                    print(f"[WIN] {current_term['display']} ({game.word}) - {current_term['category']}")
                elif game.is_lost():
                    print(f"[LOSS] answer was {current_term['display']} ({game.word})")
        elif key == ord("n"):
            letter = fsm.reject(now)
            if letter:
                last_rejected = letter
                last_rejected_until = now + 5.0
                message = f"Rejected {letter}. See coach tip to adjust."
                print(f"[REJECTED] {letter}")
        elif key in (ord("+"), ord("=")):
            threshold = min(1.0, threshold + THRESHOLD_STEP)
        elif key in (ord("-"), ord("_")):
            threshold = max(0.0, threshold - THRESHOLD_STEP)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()