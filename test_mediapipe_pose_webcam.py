"""Live pose detection from a webcam using MediaPipe Pose Landmarker."""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")
WEBCAM_INDEX = 0
WINDOW_NAME = "MediaPipe Pose (press 'q' to quit)"
PANEL_WIDTH = 520

PANEL_BG = (28, 28, 28)
TEXT_COLOR = (220, 220, 220)
HEADER_COLOR = (100, 200, 255)
MUTED_COLOR = (140, 140, 140)

LANDMARK_NAMES = [name.lower() for name in PoseLandmark.__members__]


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return

    print(f"Downloading model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except urllib.error.URLError as exc:
        print(f"Error: Could not download model: {exc}", file=sys.stderr)
        print(
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Download complete.")


def create_pose_landmarker(model_path: Path) -> vision.PoseLandmarker:
    """Create a PoseLandmarker, using GPU delegate when supported."""
    video_options = {
        "running_mode": vision.RunningMode.VIDEO,
        "num_poses": 1,
    }

    if sys.platform != "win32":
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=python.BaseOptions.Delegate.GPU,
                ),
                **video_options,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print("Using MediaPipe GPU delegate")
            return landmarker
        except (RuntimeError, ValueError) as exc:
            print(f"GPU delegate unavailable ({exc}); using CPU.")

    if sys.platform == "win32":
        print("MediaPipe GPU delegate is not supported on Windows; using CPU.")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        **video_options,
    )
    print("Using MediaPipe CPU")
    return vision.PoseLandmarker.create_from_options(options)


def draw_landmarks_on_image(
    bgr_image: np.ndarray,
    result: vision.PoseLandmarkerResult,
) -> np.ndarray:
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    annotated = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)


def draw_side_panel(
    panel: np.ndarray,
    result: vision.PoseLandmarkerResult,
    image_width: int,
    image_height: int,
    fps: float,
) -> None:
    panel[:] = PANEL_BG
    y = 24
    line_h = 15

    def put(text: str, color: tuple[int, int, int] = TEXT_COLOR, scale: float = 0.38) -> None:
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

    if not result.pose_landmarks:
        put("No pose detected", MUTED_COLOR)
        return

    put(
        "ID  Name               Vis     X       Y       Z",
        HEADER_COLOR,
        0.36,
    )
    put("-" * 48, MUTED_COLOR, 0.36)
    put("Z = world coords (meters)", MUTED_COLOR, 0.34)
    y += 2

    for person_idx, pose_landmarks in enumerate(result.pose_landmarks):
        world_landmarks = (
            result.pose_world_landmarks[person_idx]
            if result.pose_world_landmarks
            else None
        )

        if person_idx > 0:
            y += 6
        if len(result.pose_landmarks) > 1:
            put(f"Person #{person_idx}", HEADER_COLOR, 0.40)

        for landmark_id, landmark in enumerate(pose_landmarks):
            x_px = landmark.x * image_width
            y_px = landmark.y * image_height

            visibility = landmark.visibility
            if visibility is None:
                visibility = landmark.presence
            vis_text = f"{visibility:.2f}" if visibility is not None else "  --"

            if world_landmarks and landmark_id < len(world_landmarks):
                world = world_landmarks[landmark_id]
                z_text = f"{world.z:6.3f}"
            else:
                z_text = "    --"

            name = (
                LANDMARK_NAMES[landmark_id]
                if landmark_id < len(LANDMARK_NAMES)
                else f"kpt_{landmark_id}"
            )
            row = (
                f"{landmark_id:2d}  {name:<18} {vis_text:>5}  "
                f"{x_px:6.1f}  {y_px:6.1f}  {z_text}"
            )
            put(row, TEXT_COLOR, 0.36)

            if y > panel.shape[0] - 24:
                put("...", MUTED_COLOR, 0.36)
                return


def main() -> None:
    ensure_model(MODEL_PATH)

    landmarker = create_pose_landmarker(MODEL_PATH)

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
    frame_timestamp_ms = 0

    try:
        with landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Warning: Failed to read frame from webcam.", file=sys.stderr)
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                frame_timestamp_ms += 33

                annotated = draw_landmarks_on_image(frame, result)

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

                frame_h, frame_w = annotated.shape[:2]
                panel = np.zeros((frame_h, PANEL_WIDTH, 3), dtype=np.uint8)
                draw_side_panel(panel, result, frame_w, frame_h, fps)
                display = np.hstack([annotated, panel])

                cv2.imshow(WINDOW_NAME, display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
