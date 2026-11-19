"""Webcam ownership and predictable cleanup on initialization/read failures."""

import cv2

from gesturedeck.config import Config


class CameraError(RuntimeError):
    pass


class Webcam:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.capture = None
        self.source_size = None
        self.crop_box = None

    def __enter__(self):
        backend = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                   "msmf": cv2.CAP_MSMF}[self.config.camera_backend]
        self.capture = cv2.VideoCapture(self.config.camera_index, backend)
        try:
            if not self.capture.isOpened():
                raise CameraError(
                    f"Cannot open camera {self.config.camera_index}. Close other camera apps, "
                    "check Windows camera permissions, or try --camera-index 1 / --camera-backend dshow."
                )
            # These are requests; the camera may negotiate a different resolution.
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        except Exception:
            self.capture.release()
            raise
        return self

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise CameraError("Camera frame unavailable or camera disconnected. Reconnect and restart.")
        # Enforce the configured geometry on actual pixels, not driver metadata.
        # Crop centrally to preserve proportions before resizing; never stretch
        # a widescreen camera image into the 4:3 learned-model input.
        height, width = frame.shape[:2]
        self.source_size = (width, height)
        target_width, target_height = self.config.width, self.config.height
        crop_width, crop_height = width, height
        if width * target_height > height * target_width:
            crop_width = max(1, height * target_width // target_height)
        elif width * target_height < height * target_width:
            crop_height = max(1, width * target_height // target_width)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        self.crop_box = (left, top, crop_width, crop_height)
        if (width, height) != (target_width, target_height):
            frame = frame[top:top + crop_height, left:left + crop_width]
            interpolation = (cv2.INTER_AREA if crop_width >= target_width and crop_height >= target_height
                             else cv2.INTER_LINEAR)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        return cv2.flip(frame, 1) if self.config.mirror else frame

    def __exit__(self, *_):
        if self.capture is not None:
            self.capture.release()
