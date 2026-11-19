"""The only module allowed to send presentation keyboard events."""

import logging
from typing import Callable

from gesturedeck.mapping import Action
from gesturedeck.focus import powerpoint_slideshow_focused

LOGGER = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    pass


class SlideController:
    def __init__(self, dry_run: bool = True, press: Callable[[str], None] | None = None,
                 focused: Callable[[], bool] = powerpoint_slideshow_focused) -> None:
        self.dry_run = dry_run
        self._press = press
        self._focused = focused
        self.last_action = "No commands yet"
        if not dry_run and press is None:
            import pyautogui

            pyautogui.FAILSAFE = True
            self._press = pyautogui.press

    def execute(self, action: Action) -> bool:
        if action == Action.TOGGLE:
            return False
        key = {Action.NEXT: "right", Action.PREVIOUS: "left", Action.EXIT: "esc"}[action]
        if self.dry_run:
            self.last_action = f"DRY RUN: {action.value} ({key})"
        else:
            if not self._focused():
                self.last_action = "Blocked: focus the PowerPoint slideshow, lower hand, and retry"
                LOGGER.info(self.last_action)
                return False
            try:
                self._press(key)
            except Exception as exc:
                # Includes PyAutoGUI FailSafeException: stop, never retry a key.
                raise ControllerError(f"Keyboard control stopped: {exc}") from exc
            self.last_action = f"Sent: {action.value} ({key})"
        LOGGER.info(self.last_action)
        return True
