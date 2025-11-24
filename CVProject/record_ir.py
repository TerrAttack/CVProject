from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pyorbbecsdk as ob
from pyorbbecsdk import OBError, OBFormat


def setup_ir_y16_pipeline(width: int, height: int, fps: int) -> ob.Pipeline | None:
    pipeline = ob.Pipeline()
    config = ob.Config()

    # --- Device + sensor discovery (no processing here) ---
    try:
        dev = pipeline.get_device()
    except OBError as e:
        print("[record_orbbec_raw] No Orbbec device:", e)
        return None

    try:
        sensor_list = dev.get_sensor_list()
        count = sensor_list.get_count()
    except OBError as e:
        print("[record_orbbec_raw] Failed to get sensor list:", e)
        return None

    ir_sensor = None
    for i in range(count):
        try:
            sensor = sensor_list.get_sensor_by_index(i)
            profiles = sensor.get_stream_profile_list()
        except OBError:
            continue

        # We explicitly request Y16, no 8-bit, no mapping
        try:
            profiles.get_video_stream_profile(width, height, OBFormat.Y16, fps)
            ir_sensor = sensor
            print(f"[record_orbbec_raw] IR sensor at index {i} (Y16).")
            break
        except OBError:
            continue

    if ir_sensor is None:
        print("[record_orbbec_raw] No Y16 IR-capable sensor found.")
        return None

    try:
        profiles = ir_sensor.get_stream_profile_list()
        ir_profile = profiles.get_video_stream_profile(width, height, OBFormat.Y16, fps)
    except OBError as e:
        print("[record_orbbec_raw] Could not get Y16 profile:", e)
        return None

    config.enable_stream(ir_profile)

    try:
        pipeline.start(config)
        print(f"[record_orbbec_raw] Started IR Y16 stream {width}x{height}@{fps}")
    except OBError as e:
        print("[record_orbbec_raw] Failed to start pipeline:", e)
        return None

    return pipeline


def main():
    width, height, fps = 640, 576, 30
    out_dir = Path("data/raw_ir")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[record_orbbec_raw] Output folder: {out_dir.resolve()}")
    print("[record_orbbec_raw] Press q or ESC to stop.")

    pipeline = setup_ir_y16_pipeline(width, height, fps)
    if pipeline is None:
        return

    cv.namedWindow("IR RAW Preview", cv.WINDOW_NORMAL)

    frame_idx = 0
    start = time.time()

    try:
        while True:
            try:
                frame_set = pipeline.wait_for_frames(1000)
            except OBError as e:
                print("[record_orbbec_raw] Error waiting for frames:", e)
                break

            ir_frame = frame_set.get_ir_frame()
            if ir_frame is None:
                print("[record_orbbec_raw] No IR frame.")
                continue

            w = ir_frame.get_width()
            h = ir_frame.get_height()
            fmt = ir_frame.get_format()

            if fmt != OBFormat.Y16:
                print(f"[record_orbbec_raw] Unexpected IR format: {fmt}, expected Y16.")
                break

            data16 = np.frombuffer(ir_frame.get_data(), np.uint16).reshape(h, w)

            out_path = out_dir / f"frame_{frame_idx:06d}.png"
            cv.imwrite(str(out_path), data16)

            disp8 = (data16 >> 8).astype(np.uint8)
            cv.imshow("IR RAW Preview", disp8)

            frame_idx += 1

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[record_orbbec_raw] Stop requested.")
                break

    finally:
        try:
            pipeline.stop()
        except OBError:
            pass
        cv.destroyAllWindows()

    dt = time.time() - start
    if dt <= 0:
        dt = 1e-6
    print(f"[record_orbbec_raw] Saved {frame_idx} frames (~{frame_idx/dt:.1f} fps).")


if __name__ == "__main__":
    main()