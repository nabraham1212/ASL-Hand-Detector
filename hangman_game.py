import random

# Default words used when no word is passed in. Kept letter-only and
# Q-free on purpose (see note about the quit key in the runner).
# CS / cybersecurity themed to fit the project.
DEFAULT_WORDS = [
    "PYTHON", "ROBOT", "LAPTOP", "ARRAY", "DEBUG",
    "CIPHER", "BINARY", "NETWORK", "GESTURE", "FIREWALL",
]


class HangmanGame:
    """Pure Hangman logic. No webcam, no ASL, no printing.

    Feed it letters with guess_letter() and read state with get_status().
    The same guess_letter() will later accept letters from the ASL model.
    """

    def __init__(self, word=None, word_list=None, max_lives=6):
        self.max_lives = max_lives
        # Store the list so reset() can pick a fresh random word later.
        self.word_list = word_list if word_list else DEFAULT_WORDS

        self.guessed_letters = []   # every letter tried (correct + wrong), in order
        self.wrong_guesses = []     # only the letters that were NOT in the word
        self.word = ""              # set by _set_word below

        self._set_word(word)

    # ---------- internal helper ----------
    def _set_word(self, word):
        if word:
            self.word = word.strip().upper()
        else:
            self.word = random.choice(self.word_list).strip().upper()

    # ---------- core action ----------
    def guess_letter(self, letter):
        """Process one guess and return a result dictionary."""
        letter = str(letter).strip().upper()

        # Reject anything that isn't a single A-Z letter.
        if len(letter) != 1 or not letter.isalpha():
            return {
                "valid": False,
                "letter": letter,
                "correct": None,
                "repeated": False,
                "message": "Please enter a single letter A-Z.",
            }

        # Don't allow guesses after the game is over.
        if self.is_won() or self.is_lost():
            return {
                "valid": False,
                "letter": letter,
                "correct": None,
                "repeated": False,
                "message": "Game is already over. Call reset() to play again.",
            }

        # Repeated guess.
        if letter in self.guessed_letters:
            return {
                "valid": False,
                "letter": letter,
                "correct": None,
                "repeated": True,
                "message": f"You already guessed {letter}.",
            }

        # New guess: record it.
        self.guessed_letters.append(letter)

        if letter in self.word:
            return {
                "valid": True,
                "letter": letter,
                "correct": True,
                "repeated": False,
                "message": f"Correct! {letter} is in the word.",
            }
        else:
            self.wrong_guesses.append(letter)
            return {
                "valid": True,
                "letter": letter,
                "correct": False,
                "repeated": False,
                "message": f"Sorry, {letter} is not in the word. (-1 life)",
            }

    # ---------- state readers ----------
    @property
    def lives_remaining(self):
        return self.max_lives - len(self.wrong_guesses)

    def get_masked_word(self):
        """Return the word with un-guessed letters hidden, e.g. '_ P P _ E'."""
        shown = [ch if ch in self.guessed_letters else "_" for ch in self.word]
        return " ".join(shown)

    def is_won(self):
        # Won when every letter of the word has been guessed.
        return all(ch in self.guessed_letters for ch in self.word)

    def is_lost(self):
        return self.lives_remaining <= 0

    def get_status(self):
        """One bundle of state. This is what OpenCV will read later."""
        return {
            "word": self.word,                       # secret word (caller decides when to show)
            "masked_word": self.get_masked_word(),
            "guessed_letters": list(self.guessed_letters),
            "wrong_guesses": list(self.wrong_guesses),
            "lives_remaining": self.lives_remaining,
            "max_lives": self.max_lives,
            "won": self.is_won(),
            "lost": self.is_lost(),
        }

    # ---------- restart ----------
    def reset(self, word=None):
        """Clear all progress and start a new word."""
        self.guessed_letters = []
        self.wrong_guesses = []
        self._set_word(word)
        return self.get_status()