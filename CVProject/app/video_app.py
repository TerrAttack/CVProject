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
        show_debug: bool = True,
    ):
        self.capture = capture
        self.processor = processor
        self.strategy = strategy
        self.win = win
        self.show_debug = show_debug

        self.paused = False
        self.step = False

    def run(self):
        if not self.capture.is_opened():
            print("[VideoApp] Capture not opened, exiting.")
            return
        
        cv.namedWindow(self.win, cv.WINDOW_NORMAL)

        last_time = time.time()
        debug_windows = set()
        fps = 0.0

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
                if not ok:
                    break

                data = self.processor.process(frame, dt)
                out = self.strategy.process_frame(data, dt)

                vis = out.vis
                if vis is not None:
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

                # debug windows
                if self.show_debug:
                    for name, img in out.debug.items():
                        wname = f"{self.win}:{name}"
                        cv.imshow(wname, img)
                        debug_windows.add(wname)
                else:
                    for w in debug_windows:
                        cv.destroyWindow(w)
                    debug_windows.clear()

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("p"), ord(" ")):
                self.paused = not self.paused
            if key == ord("n"):
                self.step = True
            if key == ord("d"):
                self.show_debug = not self.show_debug

        self.capture.release()
        cv.destroyAllWindows()