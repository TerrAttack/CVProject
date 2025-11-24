from capture.orbbec_ir_capture import OrbbecIRCapture
from capture.cv_capture import OpenCVCapture
from detection.simple_blob_strategy import SimpleBlobStrategy
from app.video_app import VideoApp

from processing.image_processor import CompositeProcessor
from processing.stages import (
    make_scale_stage,
    make_clahe_stage,
    make_gaussian_blur_stage,
    make_otsu_threshold_stage,
)


def main():
    #capture = OrbbecIRCapture(640, 576, 30)
    capture = OpenCVCapture(0)

    if not capture.is_opened():
        print("[main] Orbbec capture failed to open. Aborting.")
        return
    
    processor = CompositeProcessor([
        make_scale_stage(1.0),
        #make_gaussian_blur_stage(10),
    ])

    strategy = SimpleBlobStrategy()

    app = VideoApp(
        capture=capture,
        processor=processor,
        strategy=strategy,
        win="IR Blob Detector",
        show_debug=True,
    )
    app.run()


if __name__ == "__main__":
    main()