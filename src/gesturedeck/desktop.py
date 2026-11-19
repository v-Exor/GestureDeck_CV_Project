"""Orange-and-white Tk desktop UI. All widget access stays on the main thread."""

from dataclasses import replace
from datetime import datetime
import logging
from queue import Empty
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk
from gesturedeck import __version__

from gesturedeck.config import Config
from gesturedeck.session import Mode, Session
from gesturedeck.theme import LIGHT

LOGGER = logging.getLogger(__name__)
PROJECT_USE = "Project Use"


class DesktopApp:
    def __init__(self, root: tk.Tk, config: Config, mode=Mode.DRY_RUN) -> None:
        self.root, self.config = root, config
        self.palm_turn_only = config.navigation_mode == "palm_turn" and not config.swipe_model_path
        self.colors = LIGHT
        self.session: Session | None = None
        self.closing = False
        self.photo = None
        self.current_frame = None
        self.settings = []
        self.mode = tk.StringVar(value=Mode(mode).value)
        self.navigation = tk.StringVar(value=(PROJECT_USE if config.navigation_mode == "palm_turn"
                                             else "Learned swipe trial" if config.swipe_model_path else "Sideways swipe"))
        self.record_enabled = tk.BooleanVar(value=bool(config.record_directory))
        self.record_slides = tk.BooleanVar(value=config.record_slideshow)
        self.camera_index = tk.StringVar(value=str(config.camera_index))
        self.backend = tk.StringVar(value=config.camera_backend)
        self.threshold = tk.StringVar(value=str(config.confidence_threshold))
        self.mirror = tk.BooleanVar(value=config.mirror)
        self.allow_keys = tk.BooleanVar(value=False)
        self.pose = tk.StringVar(value="Release")
        self.gesture = tk.StringVar(value="No hand")
        self.confidence = tk.StringVar(value="0%")
        self.hand_tracking = tk.StringVar(value="Hand tracking: waiting for a session")
        self.control = tk.StringVar(value="STOPPED")
        self.performance = tk.StringVar(value="-- FPS")
        self.latency = tk.StringVar(value="Waiting for a session")
        self.status = tk.StringVar(value="Start a session. Gesture controls activate after the countdown.")
        self.last_action = tk.StringVar(value="No commands yet")
        self.phase = tk.StringVar(value="NOT RUNNING")
        root.title(f"GestureDeck_CV {__version__} | " +
                   (PROJECT_USE if self.palm_turn_only else
                    "LEARNED SWIPE TRIAL" if config.swipe_model_path else "Presentation Control"))
        root.geometry("1240x900")
        root.minsize(1100, 850)
        root.configure(bg=LIGHT["background"])
        root.option_add("*Font", "{Segoe UI} 10")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.pause())
        root.bind("<Control-q>", lambda _: self.close())
        self._style()
        self._build()
        self._mode_changed()
        self.log("Ready. Choose a session mode, then click Start session.")
        root.after(40, self._poll)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=LIGHT["surface"],
                        background=LIGHT["surface"], foreground=LIGHT["ink"],
                        bordercolor=LIGHT["line"], padding=6)
        style.configure("Orange.Horizontal.TProgressbar", background=LIGHT["orange"],
                        troughcolor=LIGHT["orange_soft"], borderwidth=0, thickness=5)

    def label(self, parent, text=None, *, variable=None, size=10, bold=False, color="ink", bg="surface", **kwargs):
        return tk.Label(parent, text=text, textvariable=variable,
                        font=("Segoe UI", size, "bold" if bold else "normal"),
                        fg=LIGHT[color], bg=LIGHT[bg], **kwargs)

    def button(self, parent, text, command, *, primary=False, **kwargs):
        return tk.Button(parent, text=text, command=command, cursor="hand2", relief="flat",
                         bd=0, padx=14, pady=9, font=("Segoe UI", 10, "bold"),
                         bg=LIGHT["orange" if primary else "orange_soft"],
                         fg="white" if primary else LIGHT["orange"],
                         activebackground=LIGHT["orange_hover" if primary else "orange_soft"],
                         activeforeground="white" if primary else LIGHT["orange"], **kwargs)

    def card(self, parent):
        return tk.Frame(parent, bg=LIGHT["surface"], highlightbackground=LIGHT["line"], highlightthickness=1)

    def _build(self):
        sidebar = tk.Frame(self.root, bg=LIGHT["surface"], width=196,
                           highlightbackground=LIGHT["line"], highlightthickness=1)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        logo = tk.Canvas(sidebar, width=42, height=42, bg="white", highlightthickness=0)
        logo.pack(anchor="w", padx=22, pady=(28, 12))
        logo.create_rectangle(2, 2, 40, 40, fill=LIGHT["orange"], outline="")
        logo.create_line(12, 15, 28, 15, 28, 29, 12, 29, 12, 15, fill="white", width=2)
        logo.create_line(20, 30, 20, 35, fill="white", width=2)
        self.label(sidebar, "GestureDeck", size=18, bold=True).pack(anchor="w", padx=22)
        self.label(sidebar, "PRESENT WITHOUT TOUCH", size=8, color="muted").pack(anchor="w", padx=22, pady=(5, 32))
        self.label(sidebar, "WORKSPACE", size=8, color="muted", bold=True).pack(anchor="w", padx=22, pady=(0, 10))
        self.button(sidebar, "Control desk", self._show_desk).pack(fill="x", padx=12)
        self.button(sidebar, "Gesture guide", self.show_guide).pack(fill="x", padx=12, pady=8)
        self.button(sidebar, "Setup & commands", self.show_setup).pack(fill="x", padx=12)
        footer = tk.Frame(sidebar, bg="white")
        footer.pack(side="bottom", fill="x", padx=22, pady=24)
        self.label(footer, "LOCAL & ON DEVICE", size=8, bold=True, color="orange").pack(anchor="w")
        self.label(footer, "CPU inference\nLight appearance", size=9, color="muted", justify="left").pack(anchor="w", pady=8)

        main = tk.Frame(self.root, bg=LIGHT["background"])
        main.pack(side="left", fill="both", expand=True, padx=24, pady=22)
        header = tk.Frame(main, bg=LIGHT["background"])
        header.pack(fill="x", pady=(0, 12))
        self.label(header, "Presentation control", size=22, bold=True, bg="background").pack(anchor="w")
        self.label(header, "Your gestures. Your pace. Your presentation.", color="muted", bg="background").pack(anchor="w", pady=(2, 0))
        self.mode_badge = self.label(header, "DRY RUN", size=9, bold=True, color="orange", bg="orange_soft", padx=14, pady=7)
        self.mode_badge.place(relx=1, x=0, y=10, anchor="ne")

        metrics = tk.Frame(main, bg=LIGHT["background"])
        metrics.pack(fill="x", pady=(0, 16))
        for index, (title, value, detail) in enumerate([
            ("DETECTED GESTURE", self.gesture, "One hand at a time"),
            ("POSE CONFIDENCE", self.confidence, "Used for fist and Victory holds"),
            ("GESTURE CONTROL", self.control, "Hold fist to pause / resume"),
            ("PERFORMANCE", self.performance, self.latency),
        ]):
            metrics.columnconfigure(index, weight=1, uniform="metric")
            card = self.card(metrics)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 5, 0 if index == 3 else 5))
            self.label(card, title, size=8, color="muted", bold=True).pack(anchor="w", padx=14, pady=(10, 4))
            value_label = self.label(card, variable=value, size=17, bold=True)
            value_label.pack(anchor="w", padx=14)
            if index == 2:
                self.control_label = value_label
            if index == 1:
                self.confidence_bar = tk.Canvas(card, height=4, bg=LIGHT["orange_soft"], highlightthickness=0)
                self.confidence_bar.pack(fill="x", padx=14, pady=(7, 0))
            if index != 1:
                self.label(card, detail if isinstance(detail, str) else None,
                           variable=None if isinstance(detail, str) else detail, size=8, color="muted").pack(anchor="w", padx=14, pady=(4, 8))

        body = tk.Frame(main, bg=LIGHT["background"])
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        preview = self.card(body)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_header = tk.Frame(preview, bg="white")
        preview_header.pack(fill="x", padx=16, pady=12)
        self.label(preview_header, "Camera preview", bold=True, size=12).pack(side="left")
        self.label(preview_header, variable=self.phase, size=8, color="muted").pack(side="right")
        self.canvas = tk.Canvas(preview, bg=LIGHT["preview"], highlightthickness=0, height=280)
        self.canvas.pack(fill="both", expand=True, padx=12)
        self.canvas.bind("<Configure>", lambda _: self._draw_frame())
        self.label(preview, variable=self.hand_tracking, color="muted", size=9,
                   anchor="w").pack(fill="x", padx=16, pady=(5, 0))
        self.label(preview, variable=self.status, color="muted", size=9, justify="left",
                   wraplength=540, anchor="w").pack(fill="x", padx=16, pady=10)

        controls = self.card(body)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.configure(width=282)
        panel = tk.Frame(controls, bg="white", padx=16, pady=14)
        panel.pack(fill="both", expand=True)
        self.label(panel, "Session setup", size=13, bold=True).pack(anchor="w")
        self.label(panel, "Start safely. Present confidently.", size=9, color="muted").pack(anchor="w", pady=(3, 12))
        self.mode_box = self._field(panel, "Session mode", self.mode,
                                   [mode.value for mode in Mode if not self.palm_turn_only or mode != Mode.DEMO],
                                   self._mode_changed)
        self.navigation_box = self._field(panel, "Navigation gesture", self.navigation,
                                         [PROJECT_USE] if self.palm_turn_only else
                                         ["Sideways swipe", "Learned swipe trial", PROJECT_USE])
        self.camera_summary = self.label(panel, f"Camera {self.config.camera_index}  /  threshold {self.config.confidence_threshold:.0%}",
                                         color="muted", size=9)
        self.camera_summary.pack(anchor="w", pady=(10, 5))
        self.camera_settings_button = self.button(panel, "Camera & recognition settings", self.show_camera_settings)
        self.camera_settings_button.pack(fill="x", pady=(0, 9))
        self.settings.append((self.camera_settings_button, "normal"))
        self.live_check = tk.Checkbutton(panel, text="Allow slide keys this session", variable=self.allow_keys,
                                         bg="white", fg=LIGHT["danger"], activebackground="white", selectcolor="white",
                                         font=("Segoe UI", 9), anchor="w", wraplength=240, justify="left")
        self.live_check.pack(fill="x", pady=(0, 5))
        self.record_check = tk.Checkbutton(panel, text="Record webcam + landmarks locally",
                                          variable=self.record_enabled, command=self._recording_changed,
                                          bg="white", activebackground="white",
                                          font=("Segoe UI", 9), anchor="w", wraplength=240, justify="left")
        self.record_check.pack(fill="x", pady=(0, 5))
        self.settings.append((self.record_check, "normal"))
        self.slide_record_check = tk.Checkbutton(panel, text="Include PowerPoint slide snapshots",
                                                variable=self.record_slides, bg="white", activebackground="white",
                                                font=("Segoe UI", 9), anchor="w", wraplength=240, justify="left",
                                                state="normal" if self.record_enabled.get() else "disabled")
        self.slide_record_check.pack(fill="x", pady=(0, 5))
        self.mode_description = self.label(panel, "", size=9, color="muted", wraplength=235, justify="left", anchor="w")
        self.mode_description.pack(fill="x", pady=(2, 12))
        self.start_button = self.button(panel, "Start session", self.start, primary=True)
        self.start_button.pack(fill="x")
        self.stop_button = self.button(panel, "Stop session", self.stop, state="disabled")
        self.stop_button.pack(fill="x", pady=7)
        self.pause_button = self.button(panel, "Pause control   /   Esc", self.pause, state="disabled")
        self.pause_button.pack(fill="x")

        self.demo_panel = self.card(main)
        self.label(self.demo_panel, "SIMULATED INPUT / Swipes play once. Lower hand between control poses.",
                   size=9, bold=True, color="orange").pack(anchor="w", padx=12, pady=(5, 2))
        poses = tk.Frame(self.demo_panel, bg="white")
        poses.pack(fill="x", padx=10, pady=(0, 5))
        for name in ("Release", "Swipe left", "Swipe right", "Open palm", "Closed fist", "Victory", "Low confidence"):
            tk.Radiobutton(poses, text=name, value=name, variable=self.pose, command=self._demo_pose,
                           indicatoron=False, bg="white", fg=LIGHT["ink"], selectcolor=LIGHT["orange_soft"],
                           activebackground=LIGHT["orange_soft"], relief="flat", padx=9, pady=3,
                           font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True)

        activity = self.card(main)
        activity.pack(fill="x", pady=(14, 0))
        activity_header = tk.Frame(activity, bg="white")
        activity_header.pack(fill="x", padx=14, pady=(8, 4))
        self.label(activity_header, "Activity", bold=True).pack(side="left")
        self.label(activity_header, variable=self.last_action, size=9, color="orange").pack(side="right")
        self.activity = tk.Text(activity, height=2, bg="white", fg=LIGHT["muted"], relief="flat",
                                state="disabled", font=("Consolas", 9), padx=14, pady=4, wrap="word")
        self.activity.pack(fill="x", pady=(0, 5))
        self.activity.bind("<MouseWheel>", lambda event: self.activity.yview_scroll(-int(event.delta / 120), "units"))
        self.bottom = self.label(main, "Dry-run by default. No presentation keys are sent.",
                                 size=9, color="muted", bg="background", anchor="w")
        self.bottom.pack(fill="x", pady=(10, 0))

    def _field(self, parent, label, variable, options, callback=None):
        self.label(parent, label, size=9, color="muted").pack(anchor="w", pady=(2, 4))
        box = ttk.Combobox(parent, textvariable=variable, values=options, state="readonly", width=26)
        box.pack(fill="x")
        self.settings.append((box, "readonly"))
        if callback:
            box.bind("<<ComboboxSelected>>", lambda _: callback())
        return box

    def _mode_changed(self):
        mode = Mode(self.mode.get())
        self.allow_keys.set(False)
        descriptions = {
            Mode.DRY_RUN: "Recognize gestures and log commands. No real keys are sent.",
            Mode.CAMERA: "Check your webcam and mirror setting. Recognition is off.",
            Mode.DEMO: "Test swipes and control poses with simulated input. No camera or real keys.",
            Mode.LIVE: f"Controls the focused PowerPoint slideshow after a {self.config.countdown_seconds}s countdown. Focus your slideshow.",
        }
        self.mode_description.configure(text=descriptions[mode])
        self.mode_badge.configure(text={Mode.DRY_RUN: "DRY RUN", Mode.CAMERA: "CAMERA ONLY",
                                        Mode.DEMO: "DEMO / SIMULATED", Mode.LIVE: "LIVE KEYS"}[mode])
        self.live_check.configure(state="normal" if mode == Mode.LIVE else "disabled")
        self.bottom.configure(text=("Live mode: only the focused PowerPoint slideshow accepts keys. Fist pauses; Victory exits."
                                    if mode == Mode.LIVE else "Dry-run by default. No presentation keys are sent."))
        self.demo_panel.pack_forget()
        if mode == Mode.DEMO:
            # Place the demo strip just before the activity card.
            self.demo_panel.pack(fill="x", pady=(12, 0), before=self.activity.master)

    def _show_desk(self):
        self.root.focus_set()

    def log(self, text):
        self.activity.configure(state="normal")
        self.activity.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        if int(self.activity.index("end-1c").split(".")[0]) > 200:
            self.activity.delete("1.0", "2.0")
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def _set_running(self, running):
        for widget, idle_state in self.settings:
            widget.configure(state="disabled" if running else idle_state)
        self.live_check.configure(state="disabled" if running or Mode(self.mode.get()) != Mode.LIVE else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.pause_button.configure(state="normal" if running else "disabled")
        self.slide_record_check.configure(state="disabled" if running or not self.record_enabled.get() else "normal")

    def _recording_changed(self):
        if not self.record_enabled.get():
            self.record_slides.set(False)
        self.slide_record_check.configure(state="normal" if self.record_enabled.get() and self.session is None else "disabled")

    def start(self):
        if self.session is not None:
            return
        try:
            mode = Mode(self.mode.get())
            if self.palm_turn_only and (self.navigation.get() != PROJECT_USE or mode == Mode.DEMO):
                raise ValueError("This presentation launcher uses Project Use with real webcam input.")
            if mode == Mode.LIVE and not self.allow_keys.get():
                raise ValueError("Select 'Allow slide keys this session' to explicitly enable live mode.")
            config = replace(self.config, camera_index=int(self.camera_index.get()),
                             confidence_threshold=float(self.threshold.get()), camera_backend=self.backend.get(),
                             mirror=self.mirror.get(),
                             navigation_mode=("palm_turn" if self.navigation.get() == PROJECT_USE else "swipe"),
                             swipe_model_path=((self.config.swipe_model_path or "models/swipe_motion_trial.npz")
                                               if self.navigation.get().startswith("Learned") else ""),
                             swipe_trial=self.navigation.get().startswith("Learned"),
                             record_directory=((self.config.record_directory or "recordings")
                                               if self.record_enabled.get() else ""),
                             record_slideshow=self.record_enabled.get() and self.record_slides.get())
            if mode == Mode.DEMO and (config.record_directory or config.navigation_mode == "palm_turn" or config.swipe_model_path):
                raise ValueError("Learned swipes, palm turns and recording require a real webcam session, not simulated input.")
        except ValueError as exc:
            messagebox.showerror("Check session settings", str(exc), parent=self.root)
            return
        self.current_frame = None
        self.control.set("STARTING")
        self.last_action.set("No commands yet")
        self.phase.set("STARTING")
        self.status.set("Opening session. Keep your hand out of view until ready.")
        self.pose.set("Release")
        self.session = Session(config, mode)
        self.root.title(f"GestureDeck_CV {__version__} | " +
                        ("LEARNED SWIPE TRIAL" if config.swipe_model_path else
                         PROJECT_USE if config.navigation_mode == "palm_turn" else "ORIGINAL SWIPE RULES"))
        self._set_running(True)
        self.session.start()
        self.log(f"Opening {mode.value}...")
        if config.navigation_mode == "swipe" and config.swipe_model_path:
            self.log("EXPERIMENTAL learned swipe model: IPN HandS motion sequences. "
                     "Classifier scores are not measured accuracy. Test in dry run first.")
        if config.navigation_mode == "palm_turn":
            self.log("RIGHT HAND: palm to back = next; back to palm = previous. " + (
                "Opposite turns work immediately; repeat protection lasts 1s."
                if config.turn_rearm_policy == "same_direction" else "Wait 1s or lower hand between commands."))
        if config.record_directory:
            self.bottom.configure(text=("RECORDING webcam + landmarks + PowerPoint snapshots locally. Stop session to finalize files."
                                        if config.record_slideshow else
                                        "RECORDING webcam + landmarks locally. No screen/audio. Stop session to finalize files."))

    def stop(self):
        if self.session:
            self.session.stop()
            self.status.set("Stopping; waiting for camera/model cleanup...")
            self.control.set("PAUSED")
            self.stop_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")

    def pause(self):
        if self.session:
            self.session.pause()
            self.control.set("PAUSED")
            self.status.set("Pause requested. Lower hand, then hold a fist to resume.")

    def _demo_pose(self):
        if self.session and self.session.mode == Mode.DEMO:
            if self.pose.get() in ("Swipe left", "Swipe right"):
                self.session.set_demo_swipe("next" if self.pose.get() == "Swipe left" else "previous")
                return
            label = {"Release": None, "Open palm": "Open_Palm", "Victory": "Victory",
                     "Closed fist": "Closed_Fist", "Low confidence": "Open_Palm"}[self.pose.get()]
            self.session.set_demo(label, self.pose.get() == "Low confidence")

    def _poll(self):
        if self.session:
            try:
                while True:
                    self.log(self.session.events.get_nowait())
            except Empty:
                pass
            try:
                snapshot = self.session.frames.get_nowait()
            except Empty:
                snapshot = None
            if snapshot is not None:
                self.current_frame = snapshot.frame
                label = snapshot.observation.gesture or ("Unknown" if snapshot.observation.hand_present else "No hand")
                self.gesture.set(label.replace("_", " "))
                score = snapshot.observation.confidence if snapshot.observation.hand_present else 0
                self.confidence.set(f"{score:.0%}")
                tracked = snapshot.observation
                self.hand_tracking.set("Hand tracking: no hand visible" if not tracked.hand_present else
                                       f"Hand tracked / {tracked.extended_fingers}/4 long fingers extended (thumb excluded) / "
                                       f"{'open hand detected' if tracked.open_hand else 'show an open hand to swipe'}")
                if self.session.config.navigation_mode == "palm_turn" and tracked.hand_present:
                    facing = tracked.palm_facing
                    side = ("PALM" if facing > 0 else "BACK") if facing is not None and abs(facing) >= self.session.config.turn_min_facing else "EDGE / UNCERTAIN"
                    self.hand_tracking.set(f"Right-hand turn: {side} / {tracked.extended_fingers}/4 long fingers extended (thumb excluded)")
                elif self.session.config.swipe_model_path and tracked.hand_present:
                    self.hand_tracking.set("Hand tracked / experimental learned motion / finger count is not required")
                self.confidence_bar.delete("fill")
                self.confidence_bar.create_rectangle(0, 0, self.confidence_bar.winfo_width() * score, 4,
                                                     fill=LIGHT["orange"], outline="", tags="fill")
                active = snapshot.active and not self.session.stopped.is_set() and not self.session.pause_requested.is_set()
                self.control.set("PREVIEW ONLY" if self.session.mode == Mode.CAMERA else
                                 "COUNTDOWN" if snapshot.phase.startswith("Focus slideshow") else
                                 "ACTIVE" if active else "PAUSED")
                self.control_label.configure(fg=LIGHT["green" if active else "ink"])
                self.performance.set(f"{snapshot.fps:.1f} FPS" if snapshot.fps else "-- FPS")
                self.latency.set("Simulated frames" if self.session.mode == Mode.DEMO else
                                 f"{snapshot.inference_ms:.1f} ms inference" if snapshot.inference_ms is not None else "Recognition off")
                self.phase.set(snapshot.phase.upper())
                self.status.set(snapshot.status)
                self.last_action.set(snapshot.last_action)
                self._draw_frame()
            if self.session.thread is not None and not self.session.thread.is_alive():
                self.session.thread.join(timeout=0)
                self.session = None
                self._set_running(False)
                self.allow_keys.set(False)
                self.control.set("STOPPED")
                self.control_label.configure(fg=LIGHT["ink"])
                if self.phase.get() != "ERROR":
                    self.phase.set("STOPPED")
                    self.status.set("Session stopped. Start again when ready.")
                if self.closing:
                    self.root.destroy()
                    return
        if self.closing and self.session is None:
            self.root.destroy()
            return
        self.root.after(40, self._poll)

    def _draw_frame(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        self.canvas.delete("all")
        if self.current_frame is not None:
            image = Image.fromarray(self.current_frame[:, :, ::-1])
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self.canvas.create_image(width / 2, height / 2, image=self.photo)
            return
        cx, cy = width / 2, height / 2 - 22
        self.canvas.create_oval(cx-58, cy-58, cx+58, cy+58, fill=LIGHT["orange_soft"], outline="")
        self.canvas.create_rectangle(cx-28, cy-19, cx+19, cy+20, outline=LIGHT["orange"], width=3)
        self.canvas.create_polygon(cx+19, cy-9, cx+34, cy-18, cx+34, cy+18, cx+19, cy+9,
                                   fill=LIGHT["orange"], outline="")
        self.canvas.create_text(cx, cy+88, text="Ready when you are", fill=LIGHT["ink"], font=("Segoe UI", 17, "bold"))
        self.canvas.create_text(cx, cy+116, text=("Start the camera to use palm / back turns."
                                               if self.palm_turn_only else
                                               "Start the camera, or explore without one in demo mode."),
                                fill=LIGHT["muted"], font=("Segoe UI", 10), width=max(100, width-40))

    def show_guide(self):
        if self.navigation.get().startswith("Learned"):
            self._help_window("Gesture guide", "Experimental learned swipes",
                              "A supervised model reads a short sequence of hand landmarks.\n\n"
                              "Move your hand RIGHT to LEFT: next slide\n"
                              "Move your hand LEFT to RIGHT: previous slide\n\n"
                              "Keep the whole hand visible. Finger count is not required.\n"
                              "Lower your hand between attempts. Start with a dry-run session.\n"
                              "Held fist pauses/resumes; held Victory exits the slideshow.\n\n"
                              "Dataset test results do not establish accuracy on your webcam.")
            return
        if self.navigation.get() == PROJECT_USE:
            self._help_window("Gesture guide", "Turn your open RIGHT hand",
                              "Keep fingers up and spread; face your hand toward the camera.\n\n"
                              "PALM to BACK: next slide\n\nBACK to PALM: previous slide\n\n"
                              "Show each side clearly with fingers open; confirmation is brief.\n"
                              "The first side shown establishes the starting position.\n"
                              "Holding still does nothing. Lower your hand to reset.\n" +
                              ("Confirmed opposite turns work immediately.\n"
                               "Repeat protection: one second OR lower your hand.\n" if self.config.turn_rearm_policy == "same_direction" else
                               "Between commands, wait one second OR lower your hand.\n"
                               "A turn completed during that pause is ignored, not queued.\n") +
                              "Turning back also sends previous; reset out of view to repeat next.\n\n"
                              f"CLOSED FIST {self.config.pause_hold_seconds:g}s: pause / resume\n"
                              f"VICTORY {self.config.exit_hold_seconds:g}s: exit slideshow\n\n"
                              "Experimental: edge-on tracking may interrupt a turn.\n"
                              "Recording saves webcam + landmarks locally; no screen/audio.\n"
                              "Stop session to finalize recordings. Predictions need manual labeling.")
            return
        self._help_window("Gesture guide", "Swipe through your presentation",
                          "Open-hand swipe RIGHT to LEFT: next slide\n\n"
                          "Open-hand swipe LEFT to RIGHT: previous slide\n\n"
                          f"Hold CLOSED FIST {self.config.pause_hold_seconds:g}s: pause / resume once\n\n"
                          f"Hold VICTORY {self.config.exit_hold_seconds:g}s: exit slideshow and pause\n\n"
                          "A stationary open palm does nothing.\n"
                          f"Lower hand for at least {self.config.release_seconds:g}s between attempts.\n"
                          "Controls activate after countdown. Escape pauses.\n"
                          "Live tones: rising = active; falling = paused; low = blocked/exit.")

    def show_camera_settings(self):
        if self.session is not None:
            return
        popup = tk.Toplevel(self.root)
        popup.title("GestureDeck | Camera settings")
        popup.configure(bg="white")
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()
        panel = tk.Frame(popup, bg="white", padx=26, pady=22)
        panel.pack(fill="both", expand=True)
        self.label(panel, "Camera & recognition", size=18, bold=True).pack(anchor="w", pady=(0, 18))
        index = tk.StringVar(value=self.camera_index.get())
        threshold = tk.StringVar(value=self.threshold.get())
        backend = tk.StringVar(value=self.backend.get())
        mirror = tk.BooleanVar(value=self.mirror.get())
        for title, variable, start, end, increment in (
            ("Camera index", index, 0, 20, 1), ("Confidence threshold", threshold, 0.1, 1, 0.05)
        ):
            self.label(panel, title, color="muted").pack(anchor="w", pady=(4, 4))
            tk.Spinbox(panel, from_=start, to=end, increment=increment, textvariable=variable,
                       width=30, font=("Segoe UI", 11)).pack(fill="x", pady=(0, 10))
        self.label(panel, "Camera backend", color="muted").pack(anchor="w", pady=(4, 4))
        ttk.Combobox(panel, textvariable=backend, values=["auto", "dshow", "msmf"], state="readonly").pack(fill="x")
        tk.Checkbutton(panel, text="Mirror preview", variable=mirror, bg="white", activebackground="white").pack(anchor="w", pady=14)
        error = self.label(panel, "", color="danger", wraplength=300, justify="left")
        error.pack(anchor="w")

        def save():
            try:
                checked = replace(self.config, camera_index=int(index.get()),
                                  confidence_threshold=float(threshold.get()), camera_backend=backend.get(), mirror=mirror.get())
            except ValueError as exc:
                error.configure(text=str(exc))
                return
            self.camera_index.set(str(checked.camera_index))
            self.threshold.set(str(checked.confidence_threshold))
            self.backend.set(checked.camera_backend)
            self.mirror.set(checked.mirror)
            self.camera_summary.configure(text=f"Camera {checked.camera_index}  /  threshold {checked.confidence_threshold:.0%}")
            self.log("Camera and recognition settings updated for the next session.")
            popup.destroy()

        self.button(panel, "Apply settings", save, primary=True).pack(fill="x", pady=(12, 0))

    def show_setup(self):
        if self.palm_turn_only:
            self._help_window("Presentation setup", PROJECT_USE,
                              "Open your PowerPoint deck in slideshow mode.\n\n"
                              "For a recognition check, choose Recognition / dry run.\n"
                              "For slide control, choose PowerPoint / live keys,\n"
                              "then tick Allow slide keys this session.\n\n"
                              "Click Start session and focus the slideshow\n"
                              "during the eight-second countdown.\n\n"
                              "PALM to BACK: next. BACK to PALM: previous.\n"
                              "Hold a fist to pause / resume; Victory exits.\n\n"
                              "Recording is optional and starts off.\n"
                              "Click Stop session to release the camera.")
            return
        self._help_window("Setup & commands", "Run it from your terminal", 
                          "From the project folder, with .venv activated:\n\n"
                          "python -m gesturedeck --gui\n"
                          "python -m gesturedeck --demo\n"
                          "python -m gesturedeck --gui --camera-only\n"
                          "python -m gesturedeck --dry-run\n"
                          "python -m pytest\n\n"
                          "For real control, choose PowerPoint / live keys,\n"
                          "explicitly allow keys, and start the session.\n"
                          "Focus your slideshow during the countdown.\n\n"
                          "Keys require the focused PowerPoint slideshow. PowerPoint is never closed by a gesture.\n"
                          "Ctrl+Q closes this window. Escape pauses control.\n"
                          "Dark mode and executable packaging can be added later.")

    def _help_window(self, title, heading, text):
        self.pause()
        popup = tk.Toplevel(self.root)
        popup.title(f"GestureDeck | {title}")
        popup.configure(bg="white")
        popup.resizable(False, False)
        popup.transient(self.root)
        self.label(popup, heading, size=18, bold=True).pack(anchor="w", padx=26, pady=(24, 16))
        self.label(popup, text, size=11, justify="left", anchor="w").pack(padx=26, pady=(0, 18))
        self.button(popup, "Back to control desk", popup.destroy, primary=True).pack(anchor="e", padx=26, pady=(0, 24))

    def close(self):
        self.closing = True
        self.stop()
        if self.session is None:
            self.root.destroy()


def run_desktop(config: Config, mode: Mode = Mode.DRY_RUN) -> int:
    root = tk.Tk()
    app = DesktopApp(root, config, mode)
    # Periodic callbacks allow Python to handle terminal Ctrl+C during Tk's loop.
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        if app.session is not None:
            root.mainloop()
    return 0
