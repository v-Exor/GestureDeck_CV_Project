"""Headless real-model smoke check. Does not open a camera or send keys."""

import numpy as np

from gesturedeck.config import Config
from gesturedeck.recognizer import GestureRecognizer


def main() -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with GestureRecognizer(Config()) as recognizer:
        for _ in range(3):
            observation, hands = recognizer.recognize(blank)
            assert not observation.hand_present and not hands, "Unexpected hand in a black frame"
    print("PASS: official model loaded on CPU; three VIDEO frames processed; no hands detected.")


if __name__ == "__main__":
    main()
