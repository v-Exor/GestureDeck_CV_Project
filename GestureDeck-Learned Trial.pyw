"""Retained desktop launcher: palm/back turns, without a command window."""

import io
import logging
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox


def main():
    project = Path(__file__).resolve().parent
    # Anchor models and opt-in recordings to this project even from a shortcut.
    os.chdir(project)
    sys.path.insert(0, str(project / "src"))
    diagnostics = io.StringIO()
    # pythonw has no console streams. Keep startup errors available for a dialog.
    sys.stdout = sys.stdout or diagnostics
    sys.stderr = sys.stderr or diagnostics
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        stream=diagnostics)
    try:
        from gesturedeck.app import main as run

        result = run([
            "--gui", "--dry-run", "--countdown", "8", "--navigation", "palm_turn",
        ])
    except Exception:
        logging.exception("GestureDeck could not start")
        result = 1
    if result:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("GestureDeck - Project Use",
                             diagnostics.getvalue()[-4000:] or "GestureDeck could not start.",
                             parent=root)
        root.destroy()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
