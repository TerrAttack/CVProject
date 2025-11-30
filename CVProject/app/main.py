from __future__ import annotations

from capture.orbbec_ir_capture import OrbbecIRCapture
from capture.cv_capture import OpenCVCapture
from capture.raw_ir_capture import RawFolderCapture

from detection.simple_blob_strategy import SimpleBlobStrategy
from detection.ccl_detection_strategy import CCLBlobStrategy
from detection.doh_detection_strategy import DoHBlobStrategy
from detection.dog_detection_strategy import DoGBlobStrategy
from detection.log_detection_strategy import LoGBlobStrategy

from app.gui_controls import PipelineController


def main():
    #capture = OrbbecIRCapture(640, 576, 30)
    #capture = OpenCVCapture("data/videos/IR_recording.mp4")
    capture = RawFolderCapture("data/raw_ir")

    if not capture.is_opened():
        print("[main] Failed to open Orbbec IR stream.")
        return

    #strategy = SimpleBlobStrategy()
    #strategy = DoGBlobStrategy()
    strategy = LoGBlobStrategy()
    #strategy = CCLBlobStrategy()
    #strategy = DoHBlobStrategy()

    controller = PipelineController(capture, strategy)
    controller.run()


if __name__ == "__main__":
    main()