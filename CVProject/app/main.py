from capture.orbbec_ir_capture import OrbbecIRCapture
from capture.cv_capture import OpenCVCapture
from processing.image_processor import BasicProcessor
from detection.simple_blob_strategy import SimpleBlobStrategy
from app.video_app import VideoApp


def main():
    #capture = OrbbecIRCapture(640, 576, 30)
    capture = OpenCVCapture("data/videos/Dots.mp4")

    if not capture.is_opened():
        print("[main] Orbbec capture failed to open. Aborting.")
        return
    
    processor = BasicProcessor(scale=1.0)
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