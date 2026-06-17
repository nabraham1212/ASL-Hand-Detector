import os
import csv
import cv2
import HandTrackingModule as htm
from asl_utils import extract_normalized_landmarks

# ---- Config ----
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "asl_landmarks.csv")

# All STATIC ASL alphabet letters. J and Z are intentionally excluded because
# they require motion (handled later with sequence modeling).
VALID_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]
MOTION_LETTERS = {"J", "Z"}

# ---- Colors (BGR) ----
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)


def build_header():
    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    return header


def ensure_csv(path):
    """Create data/ and the CSV with a header row if they don't exist.
    Never overwrites an existing file. Returns True if it was just created.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:        # newline="" -> no blank rows on Windows
            csv.writer(f).writerow(build_header())
        return True
    return False


def load_label_counts(path):
    """Count existing samples per supported label so the screen can show totals."""
    counts = {label: 0 for label in VALID_LABELS}
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


def save_sample(path, label, features):
    with open(path, "a", newline="") as f:             # "a" = append, never overwrite
        csv.writer(f).writerow([label] + features)


def draw_panel(img, lines, x=10, y=10, font_scale=0.6, line_height=26, pad=12, width=500):
    height = pad * 2 + line_height * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x - pad + width, y - pad + height),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (text, color) in enumerate(lines):
        ty = y + line_height * i + 18
        cv2.putText(img, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)


def main():
    created = ensure_csv(CSV_PATH)
    total_counts = load_label_counts(CSV_PATH)
    session_counts = {label: 0 for label in VALID_LABELS}

    letters_with_data = sum(1 for v in total_counts.values() if v > 0)

    print("=== SignSpell Data Collection - 24 static letters ===")
    print(f"CSV {'created' if created else 'found (appending)'}: {CSV_PATH}")
    print(f"Letters with data so far: {letters_with_data}/{len(VALID_LABELS)}")
    print("Existing totals:", {k: v for k, v in total_counts.items() if v})
    print("Controls: letter=select label | SPACE=save | TAB=draw | "
          "BACKSPACE=reset session | ESC=quit\n")

    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(maxHands=1)   # one hand only for clean data

    selected_label = "A"
    draw_landmarks = True     # ON by default so you can see your hand pose
    last_message = ""

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)   # mirror; keep the SAME hand for all collection

        img = detector.findHands(img, draw=draw_landmarks)
        raw = detector.get_raw_landmarks(0)
        hand_detected = raw is not None

        lines = [
            ("SignSpell Data Collection (24 static letters)", MAGENTA),
            ("", WHITE),
            (f"Selected Label: {selected_label}", YELLOW),
            (f"Hand Detected: {'Yes' if hand_detected else 'No'}", GREEN if hand_detected else RED),
            (f"Saved This Session: {session_counts[selected_label]}", WHITE),
            (f"Total for {selected_label}: {total_counts[selected_label]}", WHITE),
            (f"Dataset: {CSV_PATH}", GRAY),
            (f"Draw landmarks: {'ON' if draw_landmarks else 'OFF'}", GREEN if draw_landmarks else GRAY),
            ("", WHITE),
            (last_message if last_message else "", YELLOW),
            ("", WHITE),
            ("Press a letter A-Y to choose label (no J/Z)", GRAY),
            ("SPACE: save sample      TAB: toggle draw", GRAY),
            ("BACKSPACE: reset session   ESC: quit", GRAY),
        ]
        draw_panel(img, lines)

        cv2.imshow("SignSpell Data Collection", img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:            # ESC quits (Q is now a label)
            break
        elif key in (8, 127):    # BACKSPACE resets session counts (R is now a label)
            session_counts = {label: 0 for label in VALID_LABELS}
            last_message = "Session counts reset (CSV untouched)."
        elif key == 9:           # TAB toggles drawing
            draw_landmarks = not draw_landmarks
        elif key == 32:          # SPACE saves a sample
            if not hand_detected:
                last_message = "No hand detected - not saved."
            else:
                features = extract_normalized_landmarks(raw)
                if features is None:
                    last_message = "Invalid hand - not saved."
                else:
                    save_sample(CSV_PATH, selected_label, features)
                    session_counts[selected_label] += 1
                    total_counts[selected_label] += 1
                    last_message = (f"Saved {selected_label}  "
                                    f"(session {session_counts[selected_label]}, "
                                    f"total {total_counts[selected_label]})")
                    print(f"[SAVED] {selected_label}  "
                          f"session={session_counts[selected_label]}  "
                          f"total={total_counts[selected_label]}")
        elif 97 <= key <= 122:   # a-z -> select a label (letters are labels only now)
            ch = chr(key).upper()
            if ch in VALID_LABELS:
                selected_label = ch
                last_message = f"Label set to {ch}."
            elif ch in MOTION_LETTERS:
                last_message = f"{ch} needs motion - not supported yet."

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Final totals:", {k: v for k, v in total_counts.items() if v})


if __name__ == "__main__":
    main()