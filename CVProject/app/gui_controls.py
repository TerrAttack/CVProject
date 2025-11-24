# app/gui_controls.py
from __future__ import annotations

import threading
import dearpygui.dearpygui as dpg

from capture.capture_base import Capture
from capture.orbbec_ir_capture import OrbbecIRCapture
from processing.image_processor import CompositeProcessor
from processing.stages import (
    make_scale_stage,
    make_clahe_stage,
    make_gaussian_blur_stage,
    make_otsu_threshold_stage,
)
from detection.detection_strategy import DetectionStrategy
from app.video_app import VideoApp


class PipelineController:
    """
    Owns:
      - pipeline_state (for image processing)
      - CompositeProcessor building
      - VideoApp instance
      - DearPyGui controls
      - VisualRange configuration on OrbbecIRCapture
    """

    def __init__(self, capture: Capture, strategy: DetectionStrategy):
        self.capture = capture
        self.strategy = strategy

        # Ensure we actually have an OrbbecIRCapture if we want visual range
        if not isinstance(self.capture, OrbbecIRCapture):
            print("[PipelineController] WARNING: Visual range controls disabled (capture is not OrbbecIRCapture).")

        # default pipeline settings
        self.pipeline_state = {
            "scale": 1.0,

            "use_clahe": False,
            "clahe_clip": 2.0,
            "use_blur": True,
            "blur_ksize": 3,
            "use_otsu": False,
        }

        # default visual range settings (16-bit units)
        self.vr_enabled = True
        self.vr_min = 10000
        self.vr_max = 20000

        # initialise capture visual range (disabled)
        if isinstance(self.capture, OrbbecIRCapture):
            self.capture.set_visual_range(False, self.vr_min, self.vr_max)

        self.processor = self._build_processor()

        self.app = VideoApp(
            capture=self.capture,
            processor=self.processor,
            strategy=self.strategy,
            win="IR Blob Detector",
        )

    # ------------------------------------------------------------
    #   Pipeline building
    # ------------------------------------------------------------
    def _build_processor(self) -> CompositeProcessor:
        s = []

        if self.pipeline_state["scale"] != 1.0:
            s.append(make_scale_stage(self.pipeline_state["scale"]))

        if self.pipeline_state["use_clahe"]:
            s.append(
                make_clahe_stage(
                    clip_limit=self.pipeline_state["clahe_clip"],
                    tile_grid_size=(8, 8),
                )
            )

        if self.pipeline_state["use_blur"]:
            s.append(make_gaussian_blur_stage(self.pipeline_state["blur_ksize"]))

        if self.pipeline_state["use_otsu"]:
            s.append(make_otsu_threshold_stage())

        return CompositeProcessor(s)

    def _rebuild_pipeline(self):
        self.processor = self._build_processor()
        self.app.processor = self.processor
        print("[PipelineController] Updated pipeline:", self.pipeline_state)

    # ------------------------------------------------------------
    #   DearPyGui setup
    # ------------------------------------------------------------
    def _setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Pipeline Controls", width=420, height=340)

        with dpg.window(label="Preprocessing + VisualRange", width=400, height=320):
            dpg.add_text("IR VisualRange (16-bit)")

            # ---------- Visual Range ----------
            def on_vr_toggle(sender, app_data):
                self.vr_enabled = bool(app_data)
                if isinstance(self.capture, OrbbecIRCapture):
                    self.capture.set_visual_range(self.vr_enabled, self.vr_min, self.vr_max)
                print("[GUI] VisualRange enabled:", self.vr_enabled)

            dpg.add_checkbox(
                label="Use Visual Range",
                default_value=self.vr_enabled,
                callback=on_vr_toggle,
            )

            def on_vr_min(sender, app_data):
                v = int(app_data)
                if v < 0:
                    v = 0
                if v >= self.vr_max:
                    v = self.vr_max - 1
                self.vr_min = v
                if isinstance(self.capture, OrbbecIRCapture):
                    self.capture.set_visual_range(self.vr_enabled, self.vr_min, self.vr_max)
                dpg.set_value(sender, v)
                print("[GUI] VisualRange min:", self.vr_min)

            def on_vr_max(sender, app_data):
                v = int(app_data)
                if v <= self.vr_min:
                    v = self.vr_min + 1
                if v > 20000:  # arbitrary upper bound
                    v = 20000
                self.vr_max = v
                if isinstance(self.capture, OrbbecIRCapture):
                    self.capture.set_visual_range(self.vr_enabled, self.vr_min, self.vr_max)
                dpg.set_value(sender, v)
                print("[GUI] VisualRange max:", self.vr_max)

            # VisualRange sliders in 16-bit units; adjust bounds as needed
            dpg.add_slider_int(
                label="VR min (16-bit)",
                min_value=0,
                max_value=20000,
                default_value=self.vr_min,
                callback=on_vr_min,
            )
            dpg.add_slider_int(
                label="VR max (16-bit)",
                min_value=1,
                max_value=20000,
                default_value=self.vr_max,
                callback=on_vr_max,
            )

            dpg.add_separator()
            dpg.add_text("Image processing")

            # CLAHE
            def on_clahe_toggle(sender, app_data):
                self.pipeline_state["use_clahe"] = bool(app_data)
                self._rebuild_pipeline()

            dpg.add_checkbox(
                label="Use CLAHE",
                default_value=self.pipeline_state["use_clahe"],
                callback=on_clahe_toggle,
            )

            def on_clahe_clip(sender, app_data):
                self.pipeline_state["clahe_clip"] = float(app_data)
                self._rebuild_pipeline()

            dpg.add_slider_float(
                label="CLAHE clip limit",
                min_value=0.1,
                max_value=10.0,
                default_value=self.pipeline_state["clahe_clip"],
                callback=on_clahe_clip,
            )

            # Blur
            def on_blur_toggle(sender, app_data):
                self.pipeline_state["use_blur"] = bool(app_data)
                self._rebuild_pipeline()

            dpg.add_checkbox(
                label="Use Gaussian Blur",
                default_value=self.pipeline_state["use_blur"],
                callback=on_blur_toggle,
            )

            def on_blur_ksize(sender, app_data):
                k = int(app_data)
                if k % 2 == 0:
                    k += 1
                if k < 1:
                    k = 1
                self.pipeline_state["blur_ksize"] = k
                self._rebuild_pipeline()
                dpg.set_value(sender, k)

            dpg.add_slider_int(
                label="Blur kernel size",
                min_value=1,
                max_value=15,
                default_value=self.pipeline_state["blur_ksize"],
                callback=on_blur_ksize,
            )

            # Otsu
            def on_otsu_toggle(sender, app_data):
                self.pipeline_state["use_otsu"] = bool(app_data)
                self._rebuild_pipeline()

            dpg.add_checkbox(
                label="Use Otsu Threshold",
                default_value=self.pipeline_state["use_otsu"],
                callback=on_otsu_toggle,
            )

            dpg.add_separator()
            dpg.add_text("Detection")

            def on_show_blobs(sender, app_data):
                # Strategy is expected to have a 'draw' flag (SimpleBlobStrategy)
                if hasattr(self.strategy, "draw"):
                    self.strategy.draw = bool(app_data)
                    print("[GUI] Show blobs:", self.strategy.draw)
                else:
                    print("[GUI] Strategy has no 'draw' attribute")

            dpg.add_checkbox(
                label="Show blob overlay",
                default_value=True,
                callback=on_show_blobs,
            )

    # ------------------------------------------------------------
    #   Run both GUI and VideoApp
    # ------------------------------------------------------------
    def run(self):
        def video_thread():
            self.app.run()
            dpg.stop_dearpygui()

        t = threading.Thread(target=video_thread, daemon=True)
        t.start()

        self._setup_gui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
