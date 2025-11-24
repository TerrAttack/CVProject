from __future__ import annotations

from capture.orbbec_ir_capture import OrbbecIRCapture
from capture.cv_capture import OpenCVCapture

from detection.simple_blob_strategy import SimpleBlobStrategy
from app.gui_controls import PipelineController


def main():
    capture = OrbbecIRCapture(640, 576, 30)
    #capture = OpenCVCapture("data/videos/Dots.mp4")

    if not capture.is_opened():
        print("[main] Failed to open Orbbec IR stream.")
        return

    strategy = SimpleBlobStrategy()

    controller = PipelineController(capture, strategy)
    controller.run()


if __name__ == "__main__":
    main()