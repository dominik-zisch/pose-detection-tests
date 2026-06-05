"""Live pose detection from a webcam using YOLO11-pose (Ultralytics)."""

import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_NAME = "yolo11n-pose.pt"
WEBCAM_INDEX = 0
WINDOW_NAME = "YOLO11 Pose (press 'q' to quit)"
PANEL_WIDTH = 500

PANEL_BG = (28, 28, 28)
TEXT_COLOR = (220, 220, 220)
HEADER_COLOR = (100, 200, 255)
MUTED_COLOR = (140, 140, 140)

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_keypoint_names(model: YOLO) -> list[str]:
    kpt_names = getattr(model, "kpt_names", None)
    if kpt_names:
        names = kpt_names.get(0)
        if names:
            return list(names)
        first = next(iter(kpt_names.values()), None)
        if first:
            return list(first)
    return COCO_KEYPOINT_NAMES


def draw_side_panel(
    panel: np.ndarray,
    result: Any,
    keypoint_names: list[str],
    fps: float,
) -> None:
    panel[:] = PANEL_BG
    y = 24
    line_h = 18

    def put(text: str, color: tuple[int, int, int] = TEXT_COLOR, scale: float = 0.45) -> None:
        nonlocal y
        cv2.putText(
            panel,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_h

    put("Landmarks", HEADER_COLOR, 0.55)
    put(f"FPS: {fps:.1f}", MUTED_COLOR, 0.42)
    y += 4

    keypoints = result.keypoints
    if keypoints is None or keypoints.data is None or len(keypoints) == 0:
        put("No pose detected", MUTED_COLOR)
        put("Z: not available (2D model)", MUTED_COLOR, 0.38)
        return

    kpt_data = keypoints.data.cpu().numpy()
    num_keypoints = kpt_data.shape[1]

    put(
        "ID  Name               Vis     X       Y       Z",
        HEADER_COLOR,
        0.38,
    )
    put("-" * 46, MUTED_COLOR, 0.38)

    for person_idx, person_kpts in enumerate(kpt_data):
        if person_idx > 0:
            y += 6
        if len(kpt_data) > 1:
            put(f"Person #{person_idx}", HEADER_COLOR, 0.42)

        for landmark_id in range(num_keypoints):
            x, y_coord = person_kpts[landmark_id, 0], person_kpts[landmark_id, 1]
            visibility = (
                float(person_kpts[landmark_id, 2])
                if person_kpts.shape[1] >= 3
                else float("nan")
            )

            name = (
                keypoint_names[landmark_id]
                if landmark_id < len(keypoint_names)
                else f"kpt_{landmark_id}"
            )

            vis_text = f"{visibility:.2f}" if not np.isnan(visibility) else "  --"
            row = (
                f"{landmark_id:2d}  {name:<18} {vis_text:>5}  "
                f"{x:6.1f}  {y_coord:6.1f}     --"
            )
            put(row, TEXT_COLOR, 0.38)

        if y > panel.shape[0] - 30:
            put("...", MUTED_COLOR, 0.38)
            break

    if y <= panel.shape[0] - 24:
        put("Z: not available (2D model)", MUTED_COLOR, 0.36)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    keypoint_names = get_keypoint_names(model)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(
            f"Error: Could not open webcam at index {WEBCAM_INDEX}.",
            file=sys.stderr,
        )
        print(
            "Check that a camera is connected and not used by another app.",
            file=sys.stderr,
        )
        sys.exit(1)

    fps = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to read frame from webcam.", file=sys.stderr)
                break

            results = model(frame, device=device, verbose=False)
            result = results[0]
            annotated = result.plot()

            now = time.perf_counter()
            elapsed = now - prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            frame_h = annotated.shape[0]
            panel = np.zeros((frame_h, PANEL_WIDTH, 3), dtype=np.uint8)
            draw_side_panel(panel, result, keypoint_names, fps)
            display = np.hstack([annotated, panel])

            cv2.imshow(WINDOW_NAME, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
