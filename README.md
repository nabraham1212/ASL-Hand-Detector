# SignSpell: ASL Hangman Learning Game

A real-time computer vision game that teaches **two things at once**: ASL fingerspelling and cybersecurity/technology vocabulary. You play Hangman by finger-spelling letters into your webcam — the app recognizes each letter with machine learning, asks you to confirm it, and submits it as a guess. Every round is a real cyber/tech term, shown with its category and a beginner-friendly definition.

This is an **educational ASL alphabet (fingerspelling) practice tool**, not a full sign-language translator.

---

## Demo

You sign a letter; the app shows the live prediction and confidence, you hold the sign steady for a second, it asks "Confirm A?", and on **Y** the letter drops into the Hangman board. A coach panel tells you how to improve your sign, and a reference guide (press **G**) shows how to form all 24 letters. Win or lose a round and the full term, its category, and its definition are revealed.

## Features

- Real-time webcam hand tracking (MediaPipe 21 landmarks)
- ASL alphabet classification for **24 static letters** (Random Forest)
- Live prediction confidence + smoothing to reduce flicker
- Hold-and-confirm system so wrong predictions never auto-submit
- Hangman integration — confirmed letters become guesses
- **Cyber/tech vocabulary**: each word shows category + definition; full meaning revealed on win/loss
- Rule-based ASL coach with per-letter tips (extra detail for confusable letters)
- In-game paged ASL reference guide (text-only, no assets)
- Custom OpenCV UI (magenta landmarks, green connections)

## Tech Stack

- **Python 3.12**
- **OpenCV** (`opencv-python`)
- **MediaPipe** (`mediapipe`)
- **scikit-learn** — `RandomForestClassifier`
- **NumPy**, **pandas**, **joblib**
- **matplotlib** (optional, for the training confusion-matrix image)

## How It Works

1. The webcam captures your hand; MediaPipe returns 21 landmarks (x, y, z).
2. Landmarks are normalized — re-centered on the wrist and scaled by palm size — so hand position and distance from the camera don't matter. This yields 63 features.
3. A `RandomForestClassifier` predicts the letter and a confidence for each class.
4. Predictions are smoothed (majority of the last 10 frames); when one letter stays stable for ~1 second, the app asks you to confirm.
5. On confirm, the letter is submitted to the Hangman engine as a guess.

## Supported Letters

The 24 **static** ASL letters:

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

## Limitations

- **J and Z are not supported yet** — they are *motion* letters and require sequence modeling, which is future work.
- This is an educational fingerspelling tool, not a certified ASL translator.
- Works best with good, even lighting and one hand clearly in frame.
- Accuracy depends on the quality and balance of the data you collect.
- The coach is rule-based; it doesn't analyze your exact hand geometry.

## Installation

```powershell
# 1. Create a virtual environment
python -m venv .venv312

# 2. Activate it (Windows PowerShell)
.\.venv312\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

```powershell
python signspell_app.py
```

or, without activating the environment:

```powershell
.\.venv312\Scripts\python.exe signspell_app.py
```

You need a trained model at `models/asl_model.pkl`. If you don't have one, see Training below.

## Controls

| Key | Action |
|-----|--------|
| `Y` | Confirm the candidate letter and submit it as a guess |
| `N` | Reject the candidate letter |
| `C` | Clear the current prediction / candidate / history |
| `D` | Toggle hand-landmark drawing |
| `+` / `-` | Adjust the confidence threshold |
| `R` | New Hangman word |
| `T` | Toggle the coach panel |
| `G` | Open / close the ASL reference guide |
| `[` / `]` | Previous / next guide page |
| `Q` | Quit |

## Cybersecurity / Technology Vocabulary

Each round's answer is a real term across four categories — **Cybersecurity**, **Networking**, **Programming**, and **Technology** (e.g. Firewall, Phishing, Encryption, Router, Compiler, Database). During play the panel shows the **category and a definition as a clue**; on win/loss it reveals the full term name, category, and meaning. So you practice fingerspelling *and* build technical vocabulary at the same time. The term bank is validated at startup — any term using unsupported letters (J/Z) or spaces in the answer is automatically removed.

## Dataset & Training

The model is trained on hand-landmark samples you collect yourself.

```powershell
# 1. Collect data: pick a letter, hold the sign, press SPACE to save a sample.
#    Aim for ~100 samples per letter across all 24 letters.
.\.venv312\Scripts\python.exe collect_asl_data.py

# 2. Train (reads data/asl_landmarks.csv, writes models/ + a confusion matrix).
.\.venv312\Scripts\python.exe train_asl_model.py

# 3. Play.
.\.venv312\Scripts\python.exe signspell_app.py
```

Data collection and live prediction share the same normalization (`asl_utils.py`), so training and play features always match.

## Project Structure

```
HandTrackingProject/
├── signspell_app.py          # final app — run this
├── HandTrackingModule.py     # MediaPipe hand-tracking wrapper
├── hangman_game.py           # pure Hangman engine
├── asl_utils.py              # shared landmark normalization (single source of truth)
├── asl_coach.py              # rule-based coach tips (24 letters)
├── asl_reference.py          # paged text reference guide (24 letters)
│
├── collect_asl_data.py       # data collection tool
├── train_asl_model.py        # model training script
│
├── data/asl_landmarks.csv    # collected samples
├── models/
│   ├── asl_model.pkl         # trained model
│   └── asl_model_info.json   # metadata (labels, accuracy, ...)
├── experiments/              # phase-by-phase development files (history)
│
├── README.md
├── requirements.txt
├── .gitignore
└── PROJECT_SUMMARY.md
```

## Future Improvements

- Support the full alphabet, including **J and Z** with motion/sequence models.
- Smarter coach that compares your landmarks to the average shape per letter.
- Text-to-speech for confirmed letters and definitions.
- Difficulty levels and score tracking; a web version.

## Portfolio Summary

> Built **SignSpell**, a real-time ASL Hangman learning game using Python, OpenCV, MediaPipe hand landmarks, and a scikit-learn `RandomForestClassifier` (24 static letters, ~0.99 test accuracy). Implemented wrist-relative landmark normalization, live prediction with smoothing and a confidence-based hold-to-confirm state machine, an educational cybersecurity/technology vocabulary mode, a rule-based signing coach, and an in-game reference guide.

---

Built by Nevin Abraham as a computer vision + machine learning learning project.
