# SignSpell - Project Summary

## What it does

SignSpell is a real-time ASL Hangman game. You play Hangman by finger-spelling
letters in American Sign Language into your webcam instead of typing. The app
recognizes the letter with a machine learning model, asks you to confirm it, and
submits it as a guess. Every round is a real cybersecurity or technology term, so
you learn technical vocabulary while you practice fingerspelling.

## How it was built (phases)

- **Hand tracking** - an OpenCV + MediaPipe module that returns 21 hand landmarks.
- **Hangman engine** - a self-contained game class (word, guesses, lives, win/loss).
- **Data collection** - a tool that saves normalized landmark samples per letter to CSV.
- **Training** - validates the dataset, trains a `RandomForestClassifier`, reports
  accuracy + a confusion matrix, and saves the model and metadata.
- **Live prediction** - loads the model and predicts letters from the webcam.
- **Confirmation** - smoothing + a hold-to-confirm state machine so wrong, mid-motion
  predictions never get submitted.
- **Game integration** - confirmed letters become Hangman guesses.
- **Coach** - rule-based per-letter tips, with extra detail for confusable letters.
- **Reference guide** - an in-game paged guide showing how to form all 24 letters.
- **Final app + vocabulary** - branding, startup checks, and a cyber/tech term bank
  with categories and definitions.

## ASL model approach

Each detected hand is converted to 63 features: 21 landmarks x (x, y, z). The
features are made **position-invariant** (re-centered on the wrist) and
**scale-invariant** (divided by palm size), so hand location and camera distance
don't matter. A `RandomForestClassifier` predicts the letter and a per-class
confidence. The same normalization function is shared by data collection and live
prediction, which guarantees training and serving features always match - the most
important design decision in the project.

## Supported letters

The 24 static ASL letters: A B C D E F G H I K L M N O P Q R S T U V W X Y.

## Cybersecurity / technology vocabulary feature

The Hangman answers come from a validated term bank spanning Cybersecurity,
Networking, Programming, and Technology. During play the screen shows the term's
category and a definition as a clue; on win/loss it reveals the full term and its
meaning. The bank is checked at startup and any term with unsupported letters (J/Z)
or spaces in the answer is removed automatically.

## Limitations

- J and Z (motion letters) are not supported yet.
- It is an educational fingerspelling tool, not a certified ASL translator.
- Accuracy depends on lighting and the quality/balance of the collected dataset.
- The coach is rule-based and does not analyze exact hand geometry.

## Future work

- Add J/Z with motion/sequence models (e.g. an LSTM over landmark frames).
- A geometry-aware coach that compares your landmarks to the average per letter.
- Text-to-speech, score tracking, difficulty levels, and a web version.

---

*Built by Nevin Abraham - Junior, Paradise Valley High School CREST STEM Program*
