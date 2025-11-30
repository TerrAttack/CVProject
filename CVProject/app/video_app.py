from __future__ import annotations
import cv2 as cv
import time

from capture.capture_base import Capture
from processing.image_processor import ImageProcessor
from detection.detection_strategy import DetectionStrategy


class VideoApp:
    def __init__(
        self,
        capture: Capture,
        processor: ImageProcessor,
        strategy: DetectionStrategy,
        win: str = "Video",
    ):
        self.capture = capture
        self.processor = processor
        self.strategy = strategy
        self.win = win

        self.paused = False
        self.step = False

    def run(self):
        if not self.capture.is_opened():
            print("[VideoApp] Capture not opened.")
            return

        cv.namedWindow(self.win, cv.WINDOW_NORMAL)

        last_time = time.time()
        fps = 0.0

        print("[VideoApp] Keys: q/ESC=quit, space/p=pause, n=step")

        while True:
            if not self.paused or self.step:
                self.step = False

                now = time.time()
                dt = now - last_time
                last_time = now
                if dt <= 0:
                    dt = 1e-6
                fps = 1.0 / dt

                ok, frame = self.capture.read()
                if not ok or frame is None:
                    # Voor camera's is dit echt een fout.
                    # Voor je loopende sources (OpenCVCapture met file,
                    # RawFolderCapture) hoort read() altijd True te blijven geven.
                    print("[VideoApp] Capture ended or error.")
                    break

                data = self.processor.process(frame, dt)
                out = self.strategy.process_frame(data, dt)

                vis = out.vis
                if vis is None:
                    vis = data.bgr

                cv.putText(
                    vis,
                    f"FPS: {fps:.1f}",
                    (5, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

                cv.imshow(self.win, vis)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("p"), ord(" ")):
                self.paused = not self.paused
            if key == ord("n"):
                self.step = True

        self.capture.release()
        cv.destroyAllWindows()
        print("[VideoApp] Stopped.")
