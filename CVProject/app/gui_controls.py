from __future__ import annotations

import threading
import dearpygui.dearpygui as dpg

from capture.capture_base import Capture
from processing.image_processor import CompositeProcessor
from processing.stages import (
    make_gaussian_blur_stage,
    make_median_blur_stage,
    make_otsu_threshold_stage,
    make_opening_stage,
    make_closing_stage,
)
from detection.detection_strategy import DetectionStrategy
from app.video_app import VideoApp


class PipelineController:
    """
    Owns:
      - pipeline_state (for image processing)
      - CompositeProcessor building (the actual pipeline)
      - VideoApp instance
      - DearPyGui controls

    VisualRange gedrag:
      - Als capture een `set_visual_range(enabled, min, max)`-methode heeft
        (bijv. OrbbecIRCapture, RawFolderCapture), dan sturen we die met
        16-bit min/max sliders aan.
      - Voor andere captures (OpenCVCapture) bestaat VisualRange niet.
    """

    def __init__(self, capture: Capture, strategy: DetectionStrategy):
        self.capture = capture
        self.strategy = strategy

        # Detecteer of deze capture VisualRange ondersteunt via duck-typing
        self.has_visual_range = hasattr(self.capture, "set_visual_range")

        # Pipeline settings (voor alle bronnen)
        self.pipeline_state = {
            # Blur stages
            "use_gaussian_blur": True,
            "gaussian_ksize": 3,
            "use_median_blur": False,
            "median_ksize": 3,

            # Threshold
            "use_otsu": False,

            # Morphology
            "use_opening": False,
            "use_closing": False,
            "morph_ksize": 3,
        }

        # Capture-level visual range (16-bit)
        self.vr_enabled = False
        self.vr_min = 15000
        self.vr_max = 20000

        # Initialise VisualRange op captures die het ondersteunen
        if self.has_visual_range:
            self.capture.set_visual_range(
                self.vr_enabled, self.vr_min, self.vr_max
            )

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
        """
        Volgorde:
        1) Gaussian blur (optioneel)
        2) Median blur (optioneel)
        3) Otsu threshold (optioneel, overschrijft gray -> binary)
        4) Opening (optioneel)
        5) Closing (optioneel)

        Let op:
        - Opening/closing werken bij voorkeur op binary (na Otsu),
          anders op gray.
        """
        stages = []

        # 1) Gaussian blur
        if self.pipeline_state["use_gaussian_blur"]:
            stages.append(
                make_gaussian_blur_stage(
                    self.pipeline_state["gaussian_ksize"]
                )
            )

        # 2) Median blur
        if self.pipeline_state["use_median_blur"]:
            stages.append(
                make_median_blur_stage(
                    self.pipeline_state["median_ksize"]
                )
            )

        # 3) Otsu threshold
        if self.pipeline_state["use_otsu"]:
            stages.append(make_otsu_threshold_stage())

        # 4) Morphological opening
        if self.pipeline_state["use_opening"]:
            stages.append(
                make_opening_stage(
                    self.pipeline_state["morph_ksize"]
                )
            )

        # 5) Morphological closing
        if self.pipeline_state["use_closing"]:
            stages.append(
                make_closing_stage(
                    self.pipeline_state["morph_ksize"]
                )
            )

        return CompositeProcessor(stages)

    def _rebuild_pipeline(self):
        self.processor = self._build_processor()
        self.app.processor = self.processor
        print("[PipelineController] Updated pipeline:", self.pipeline_state)

    # ------------------------------------------------------------
    #   DearPyGui setup
    # ------------------------------------------------------------
    def _setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Pipeline Controls", width=440, height=480)

        with dpg.window(label="Preprocessing + Detection", width=420, height=460):
            # ---------- Visual Range (16-bit, capture-level) ----------
            if self.has_visual_range:
                dpg.add_text("IR VisualRange (16-bit window in capture)")

                def on_vr_toggle(sender, app_data):
                    self.vr_enabled = bool(app_data)
                    self.capture.set_visual_range(
                        self.vr_enabled, self.vr_min, self.vr_max
                    )
                    print("[GUI] VisualRange enabled:", self.vr_enabled)

                dpg.add_checkbox(
                    label="Use VisualRange (capture, 16-bit)",
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
                    dpg.set_value(sender, v)
                    self.capture.set_visual_range(
                        self.vr_enabled, self.vr_min, self.vr_max
                    )
                    print("[GUI] VisualRange min:", self.vr_min)

                def on_vr_max(sender, app_data):
                    v = int(app_data)
                    if v <= self.vr_min:
                        v = self.vr_min + 1
                    if v > 20000:
                        v = 20000
                    self.vr_max = v
                    dpg.set_value(sender, v)
                    self.capture.set_visual_range(
                        self.vr_enabled, self.vr_min, self.vr_max
                    )
                    print("[GUI] VisualRange max:", self.vr_max)

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
            dpg.add_text("Blur")

            # ---------- Gaussian Blur ----------
            def on_gaussian_blur_toggle(sender, app_data):
                self.pipeline_state["use_gaussian_blur"] = bool(app_data)
                self._rebuild_pipeline()
                print("[GUI] Use Gaussian Blur:",
                      self.pipeline_state["use_gaussian_blur"])

            dpg.add_checkbox(
                label="Use Gaussian Blur",
                default_value=self.pipeline_state["use_gaussian_blur"],
                callback=on_gaussian_blur_toggle,
            )

            # ---------- Median Blur ----------
            def on_median_blur_toggle(sender, app_data):
                self.pipeline_state["use_median_blur"] = bool(app_data)
                self._rebuild_pipeline()
                print("[GUI] Use Median Blur:",
                      self.pipeline_state["use_median_blur"])

            dpg.add_checkbox(
                label="Use Median Blur",
                default_value=self.pipeline_state["use_median_blur"],
                callback=on_median_blur_toggle,
            )

            # Shared kernel size voor beide blurs
            def on_blur_ksize(sender, app_data):
                k = int(app_data)
                if k % 2 == 0:
                    k += 1
                if k < 1:
                    k = 1
                self.pipeline_state["gaussian_ksize"] = k
                self.pipeline_state["median_ksize"] = k
                dpg.set_value(sender, k)
                self._rebuild_pipeline()
                print("[GUI] Blur kernel size:", k)

            dpg.add_slider_int(
                label="Blur kernel size (odd)",
                min_value=1,
                max_value=15,
                default_value=self.pipeline_state["gaussian_ksize"],
                callback=on_blur_ksize,
            )

            # ---------- Otsu ----------
            dpg.add_separator()
            dpg.add_text("Thresholding")

            def on_otsu_toggle(sender, app_data):
                self.pipeline_state["use_otsu"] = bool(app_data)
                self._rebuild_pipeline()
                print("[GUI] Use Otsu Threshold:",
                      self.pipeline_state["use_otsu"])

            dpg.add_checkbox(
                label="Use Otsu Threshold",
                default_value=self.pipeline_state["use_otsu"],
                callback=on_otsu_toggle,
            )

            # ---------- Morphology ----------
            dpg.add_separator()
            dpg.add_text("Morphology (opening/closing)")

            def on_opening_toggle(sender, app_data):
                self.pipeline_state["use_opening"] = bool(app_data)
                self._rebuild_pipeline()
                print("[GUI] Use Opening:",
                      self.pipeline_state["use_opening"])

            dpg.add_checkbox(
                label="Use Opening",
                default_value=self.pipeline_state["use_opening"],
                callback=on_opening_toggle,
            )

            def on_closing_toggle(sender, app_data):
                self.pipeline_state["use_closing"] = bool(app_data)
                self._rebuild_pipeline()
                print("[GUI] Use Closing:",
                      self.pipeline_state["use_closing"])

            dpg.add_checkbox(
                label="Use Closing",
                default_value=self.pipeline_state["use_closing"],
                callback=on_closing_toggle,
            )

            def on_morph_ksize(sender, app_data):
                k = int(app_data)
                if k % 2 == 0:
                    k += 1
                if k < 1:
                    k = 1
                self.pipeline_state["morph_ksize"] = k
                dpg.set_value(sender, k)
                self._rebuild_pipeline()
                print("[GUI] Morph kernel size:", k)

            dpg.add_slider_int(
                label="Morph kernel size (odd)",
                min_value=1,
                max_value=15,
                default_value=self.pipeline_state["morph_ksize"],
                callback=on_morph_ksize,
            )

            # ---------- Detection ----------
            dpg.add_separator()
            dpg.add_text("Detection")

            # Show blob overlay toggle (strategy-side)
            def on_show_blobs(sender, app_data):
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
