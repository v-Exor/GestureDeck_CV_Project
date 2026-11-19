"""Model category names to application events; no keyboard or CV imports."""

from enum import Enum


class Action(str, Enum):
    NEXT = "next"
    PREVIOUS = "previous"
    TOGGLE = "toggle"
    EXIT = "exit"


DEFAULT_MAPPING = {
    "Closed_Fist": Action.TOGGLE,
    "Victory": Action.EXIT,
}
CANNED_GESTURES = {
    "Victory", "Closed_Fist", "Open_Palm", "Pointing_Up",
    "Thumb_Down", "Thumb_Up", "ILoveYou",
}


def action_for_gesture(gesture: str | None, mapping: dict[str, Action]) -> Action | None:
    return mapping.get(gesture) if gesture else None
