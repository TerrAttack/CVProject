Benodigd heden:
Python 3.13
Pip 25.3
Numpy 2.2.6
Opencv-python 4.12.0.88
Pyorbbecsdk2 2.0.15
Dearpygui 2.1.1

Gebruik:
in app/main.py:
uncomment one of the capture steam methods
    #capture = OrbbecIRCapture(640, 576, 30)
    #capture = OpenCVCapture("data/videos/Dots.mp4")
    #capture = RawFolderCapture("data/raw_ir")

uncomment one of the detection methods
    #strategy = SimpleBlobStrategy()
    #strategy = DoGBlobStrategy()
    #strategy = LoGBlobStrategy()
    #strategy = CCLBlobStrategy()
    #strategy = DoHBlobStrategy()

run "python -m app.main" in console
