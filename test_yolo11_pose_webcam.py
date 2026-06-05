"""Live pose detection from a webcam using YOLO11-pose (Ultralytics)."""

import sys
import time

import cv2
import torch
from ultralytics import YOLO

MODEL_NAME = "yolo11n-pose.pt"
WEBCAM_INDEX = 0
WINDOW_NAME = "YOLO11 Pose (press 'q' to quit)"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

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
            annotated = results[0].plot()

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

            cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
