from dataclasses import dataclass
from pathlib import Path
import math
import threading
import time

import cv2
import numpy as np

from hp60_sdk import HP60SDKCamera


cv2.setUseOptimized(True)

PROJECT_DIR = Path(__file__).resolve().parent
SDK_ROOT = PROJECT_DIR / "EaiCameraSdk_v1.2.28.20241015"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WINDOW_NAME = "Depth Floor Distance Simulator"
RGB_DEBUG_WINDOW = "HP60 RGB Debug"
DEPTH_DEBUG_WINDOW = "HP60 Depth Debug"

RGB_DEPTH_DEBUG = False
SHOW_FPS_COUNTER = True
MAX_PROGRAM_FPS = 30.0
FPS_COUNTER_SMOOTHING = 0.9

USE_YOLO_PERSON_DETECTION = True
YOLO_ASYNC_DETECTION = True
YOLO_MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_PERSON_CONFIDENCE = 0.30
YOLO_EVERY_FRAMES = 5
YOLO_IMAGE_SIZE = 320
YOLO_DEVICE = "cpu"
YOLO_USE_HALF_ON_CUDA = False
YOLO_TORCH_THREADS = 1
YOLO_SLOW_INFERENCE_SECONDS = 0.22
YOLO_SLOW_COOLDOWN_SECONDS = 0.65
FAST_OBJECT_CENTER_SPEED_NORM_PER_SEC = 1.05
FAST_OBJECT_DEPTH_SPEED_M_PER_SEC = 2.0
FAST_OBJECT_COOLDOWN_SECONDS = 1.25
YOLO_MIN_BOX_HEIGHT_RATIO = 0.32
YOLO_MIN_BOX_WIDTH_RATIO = 0.08
YOLO_MIN_BOX_ASPECT_RATIO = 1.15
YOLO_MAX_BOX_ASPECT_RATIO = 5.8
YOLO_DEPTH_CENTER_WIDTH_RATIO = 0.34
YOLO_DEPTH_CENTER_HEIGHT_RATIO = 0.28
YOLO_MIN_DEPTH_PIXELS = 25
MAX_TRACKED_PEOPLE = 6
TRACK_MATCH_MAX_CENTER_DIST = 0.18
TRACK_MATCH_MAX_DEPTH_DELTA_M = 0.65

ACTIVE_NEAR_M = 0.5
ACTIVE_FAR_M = 3.5
FOREGROUND_BAND_M = 0.48

# Depth-only human filter. Increase these values if chairs/tables are detected,
# decrease them if real people disappear too easily.
MIN_PERSON_AREA = 2800
MIN_PERSON_HEIGHT_PX = 85
MIN_PERSON_WIDTH_PX = 28
MIN_PERSON_ASPECT_RATIO = 0.9
MAX_PERSON_ASPECT_RATIO = 6.2
MAX_PERSON_FILL_RATIO = 1.05
STABLE_PERSON_FRAMES = 3
LOST_PERSON_GRACE_SECONDS = 0.25

FLOOR_HALF_WIDTH_M = 1.6
FLOOR_HORIZON_Y = 160
FLOOR_NEAR_Y = 650
CAMERA_SCREEN_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 54)

DEPTH_ZONES = [
    {
        "name": "NEAR",
        "label": "Energy Zone",
        "min_m": 0.5,
        "max_m": 1.3,
        "color": (45, 120, 255),
    },
    {
        "name": "MIDDLE",
        "label": "Balance Zone",
        "min_m": 1.3,
        "max_m": 2.4,
        "color": (95, 235, 145),
    },
    {
        "name": "FAR",
        "label": "Space Zone",
        "min_m": 2.4,
        "max_m": 3.5,
        "color": (255, 115, 215),
    },
]

AVATAR_TRAIL_DECAY = 0.88
AVATAR_TRAIL_WEIGHT = 0.72
AVATAR_TRAIL_SCALE = 0.5
FAST_RENDER_MODE = True


@dataclass
class PersonState:
    present: bool = False
    x_norm: float = 0.5
    y_norm: float = 0.5
    distance_m: float = 0.0
    area: float = 0.0
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    track_id: int = 0


def clone_person(person):
    return PersonState(
        present=person.present,
        x_norm=person.x_norm,
        y_norm=person.y_norm,
        distance_m=person.distance_m,
        area=person.area,
        confidence=person.confidence,
        bbox=person.bbox,
        track_id=person.track_id,
    )


def clone_people(people):
    return [clone_person(person) for person in people]


class SmoothPerson:
    def __init__(self):
        self.state = PersonState()
        self.last_seen_time = 0.0
        self.stable_seen_frames = 0

    def update(self, measured):
        now = time.time()
        if measured.present:
            self.stable_seen_frames += 1
            self.last_seen_time = now
            if self.stable_seen_frames < STABLE_PERSON_FRAMES and not self.state.present:
                return self.state
            if not self.state.present:
                self.state = measured
            else:
                alpha = 0.2
                self.state.x_norm = self.state.x_norm * (1.0 - alpha) + measured.x_norm * alpha
                self.state.y_norm = self.state.y_norm * (1.0 - alpha) + measured.y_norm * alpha
                self.state.distance_m = self.state.distance_m * (1.0 - alpha) + measured.distance_m * alpha
                self.state.area = self.state.area * (1.0 - alpha) + measured.area * alpha
                self.state.confidence = measured.confidence
                self.state.bbox = measured.bbox
                self.state.present = True
        else:
            self.stable_seen_frames = 0

        if now - self.last_seen_time > LOST_PERSON_GRACE_SECONDS:
            self.state.present = False
        return self.state


class MultiPersonSmoother:
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1

    def update(self, measured_people):
        now = time.time()
        unmatched_track_ids = set(self.tracks.keys())
        updated_track_ids = set()

        measured_people = [
            person for person in measured_people
            if person.present and ACTIVE_NEAR_M <= person.distance_m <= ACTIVE_FAR_M
        ]
        measured_people = sorted(
            measured_people,
            key=lambda person: person.confidence * max(1.0, person.area),
            reverse=True,
        )[:MAX_TRACKED_PEOPLE]

        for measured in measured_people:
            track_id = self._best_track_match(measured, unmatched_track_ids)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                state = clone_person(measured)
                state.track_id = track_id
                self.tracks[track_id] = {
                    "state": state,
                    "last_seen": now,
                    "stable_frames": 1,
                }
            else:
                unmatched_track_ids.remove(track_id)
                track = self.tracks[track_id]
                state = track["state"]
                alpha = 0.28
                state.x_norm = state.x_norm * (1.0 - alpha) + measured.x_norm * alpha
                state.y_norm = state.y_norm * (1.0 - alpha) + measured.y_norm * alpha
                state.distance_m = state.distance_m * (1.0 - alpha) + measured.distance_m * alpha
                state.area = state.area * (1.0 - alpha) + measured.area * alpha
                state.confidence = measured.confidence
                state.bbox = measured.bbox
                state.present = True
                state.track_id = track_id
                track["last_seen"] = now
                track["stable_frames"] += 1

            updated_track_ids.add(track_id)

        for track_id in list(self.tracks.keys()):
            if track_id not in updated_track_ids:
                track = self.tracks[track_id]
                track["stable_frames"] = 0
                if now - track["last_seen"] > LOST_PERSON_GRACE_SECONDS:
                    del self.tracks[track_id]

        people = []
        for track_id, track in self.tracks.items():
            if track["stable_frames"] >= STABLE_PERSON_FRAMES:
                state = track["state"]
                state.track_id = track_id
                people.append(state)

        return sorted(people, key=lambda person: person.distance_m)

    def _best_track_match(self, measured, available_track_ids):
        best_track_id = None
        best_score = None
        for track_id in available_track_ids:
            state = self.tracks[track_id]["state"]
            center_dist = math.hypot(measured.x_norm - state.x_norm, measured.y_norm - state.y_norm)
            depth_delta = abs(measured.distance_m - state.distance_m)
            if center_dist > TRACK_MATCH_MAX_CENTER_DIST:
                continue
            if depth_delta > TRACK_MATCH_MAX_DEPTH_DELTA_M:
                continue

            score = center_dist + depth_delta * 0.12
            if best_score is None or score < best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id


class FPSMeter:
    def __init__(self):
        self.last_time = time.perf_counter()
        self.fps = 0.0

    def update(self):
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now
        if elapsed <= 0:
            return self.fps

        instant_fps = 1.0 / elapsed
        if self.fps <= 0:
            self.fps = instant_fps
        else:
            self.fps = self.fps * FPS_COUNTER_SMOOTHING + instant_fps * (1.0 - FPS_COUNTER_SMOOTHING)
        return self.fps


def limit_program_fps(loop_start_time):
    if MAX_PROGRAM_FPS <= 0:
        return

    target_seconds = 1.0 / float(MAX_PROGRAM_FPS)
    remaining = target_seconds - (time.perf_counter() - loop_start_time)
    if remaining > 0:
        time.sleep(remaining)


def depth_to_meters(depth_frame):
    if depth_frame is None:
        return None

    depth = depth_frame.astype(np.float32)
    if np.issubdtype(depth_frame.dtype, np.integer):
        depth = depth / 1000.0

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None

    # Some SDK modes may provide float millimeters instead of float meters.
    if not np.issubdtype(depth_frame.dtype, np.integer) and float(np.nanmax(depth[valid])) > 20.0:
        depth = depth / 1000.0
    return depth


def prepare_rgb_for_detection(rgb_frame):
    if rgb_frame is None:
        return None

    frame = rgb_frame
    if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.flip(frame, 1)


def prepare_depth_for_detection(depth_m):
    if depth_m is None:
        return None
    if depth_m.shape[1] != CAMERA_WIDTH or depth_m.shape[0] != CAMERA_HEIGHT:
        depth_m = cv2.resize(depth_m, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return cv2.flip(depth_m, 1)


def depth_to_debug_display(depth_m):
    if depth_m is None:
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        draw_text(frame, "Waiting for depth frame", (34, CAMERA_HEIGHT // 2), 0.72)
        return frame

    if depth_m.shape[1] != CAMERA_WIDTH or depth_m.shape[0] != CAMERA_HEIGHT:
        depth_m = cv2.resize(depth_m, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_8u = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    if np.any(valid):
        depth_8u[valid] = np.clip(
            (depth_m[valid] - ACTIVE_NEAR_M) * 255.0 / (ACTIVE_FAR_M - ACTIVE_NEAR_M),
            0,
            255,
        ).astype(np.uint8)

    frame = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
    frame[~valid] = (0, 0, 0)
    return cv2.flip(frame, 1)


def rgb_to_debug_display(rgb_frame):
    if rgb_frame is None:
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        draw_text(frame, "Waiting for RGB frame", (42, CAMERA_HEIGHT // 2), 0.72)
        return frame

    frame = rgb_frame
    if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.flip(frame, 1)


def draw_debug_windows(rgb_frame, depth_m, people):
    rgb_debug = rgb_to_debug_display(rgb_frame)
    depth_debug = depth_to_debug_display(depth_m)

    for person in people:
        marker = (
            int(person.x_norm * CAMERA_WIDTH),
            int(person.y_norm * CAMERA_HEIGHT),
        )
        color = zone_color_for_distance(person.distance_m)
        if person.bbox is not None:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(rgb_debug, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.rectangle(depth_debug, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(rgb_debug, marker, 16, color, 2, cv2.LINE_AA)
        cv2.circle(depth_debug, marker, 16, color, 2, cv2.LINE_AA)
        label = f"#{person.track_id} {person.distance_m:.2f} m"
        draw_text(rgb_debug, label, (marker[0] + 18, marker[1] - 12), 0.58, color, 1)
        draw_text(depth_debug, label, (marker[0] + 18, marker[1] - 12), 0.58, color, 1)

    if people:
        draw_text(rgb_debug, f"YOLO humans {len(people)}", (16, 58), 0.56, (90, 255, 130), 1)

    draw_text(rgb_debug, "RGB debug", (16, 30), 0.62, (255, 255, 255), 1)
    draw_text(depth_debug, "Depth debug", (16, 30), 0.62, (255, 255, 255), 1)
    cv2.imshow(RGB_DEBUG_WINDOW, rgb_debug)
    cv2.imshow(DEPTH_DEBUG_WINDOW, depth_debug)


def close_debug_windows():
    for window_name in (RGB_DEBUG_WINDOW, DEPTH_DEBUG_WINDOW):
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def update_people_depth_from_boxes(people, depth_m):
    if not people:
        return []

    depth = prepare_depth_for_detection(depth_m)
    if depth is None:
        return []

    updated_people = []
    for person in people:
        if person.bbox is None:
            continue
        distance_m = median_depth_in_yolo_box(depth, person.bbox)
        if distance_m is None:
            continue

        updated = clone_person(person)
        updated.distance_m = distance_m
        updated_people.append(updated)

    return updated_people


class YOLOPersonDetector:
    def __init__(self):
        self.model = None
        self.error = ""
        self.enabled = USE_YOLO_PERSON_DETECTION
        self.frame_index = 0
        self.cached_people = []
        self.device = "cpu"
        self.use_half = False
        if self.enabled:
            self.load()

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        if not YOLO_MODEL_PATH.exists():
            self.error = f"YOLO model not found: {YOLO_MODEL_PATH.name}"
            return

        try:
            from ultralytics import YOLO

            try:
                import torch

                torch.set_num_threads(max(1, int(YOLO_TORCH_THREADS)))
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            if YOLO_DEVICE == "auto":
                import torch

                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif YOLO_DEVICE == "cpu":
                self.device = "cpu"
            else:
                self.device = YOLO_DEVICE
            self.use_half = bool(YOLO_USE_HALF_ON_CUDA and self.device != "cpu")
            self.model = YOLO(str(YOLO_MODEL_PATH))
            self.model.to(self.device)
            self.error = ""
        except Exception as exc:
            self.model = None
            self.error = f"YOLO load failed: {exc}"

    def detect(self, rgb_frame, depth_m):
        if not self.ready:
            return []

        self.frame_index += 1
        detect_interval = max(1, int(YOLO_EVERY_FRAMES))
        should_run_yolo = self.frame_index == 1 or (self.frame_index - 1) % detect_interval == 0
        if not should_run_yolo:
            return self.update_cached_depth(depth_m)

        return self.detect_frame(rgb_frame, depth_m)

    def detect_frame(self, rgb_frame, depth_m):
        self.cached_people = self.find_people(rgb_frame, depth_m)
        return clone_people(self.cached_people)

    def find_people(self, rgb_frame, depth_m):
        rgb = prepare_rgb_for_detection(rgb_frame)
        depth = prepare_depth_for_detection(depth_m)
        if rgb is None or depth is None:
            return []

        try:
            results = self.model(
                rgb,
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_PERSON_CONFIDENCE,
                classes=[YOLO_PERSON_CLASS_ID],
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
        except Exception as exc:
            self.error = f"YOLO detect failed: {exc}"
            return []

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        candidates = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if cls_id != YOLO_PERSON_CLASS_ID or confidence < YOLO_PERSON_CONFIDENCE:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            bbox = clip_bbox(int(x1), int(y1), int(x2), int(y2))
            person = person_from_yolo_box(bbox, confidence, depth)
            if person is not None:
                candidates.append(person)

        return sorted(
            candidates,
            key=lambda candidate: candidate.confidence * max(1.0, candidate.area),
            reverse=True,
        )[:MAX_TRACKED_PEOPLE]

    def update_cached_depth(self, depth_m):
        self.cached_people = update_people_depth_from_boxes(self.cached_people, depth_m)
        return clone_people(self.cached_people)


class AsyncYOLOPersonDetector:
    def __init__(self):
        self.detector = YOLOPersonDetector()
        self.frame_index = 0
        self.latest_people = []
        self.last_result_time = time.perf_counter()
        self.cooldown_until = 0.0
        self.pending = False
        self.lock = threading.Lock()
        self.worker = None

    @property
    def ready(self):
        return self.detector.ready

    @property
    def error(self):
        return self.detector.error

    def detect(self, rgb_frame, depth_m):
        if not self.ready:
            return []

        self.frame_index += 1
        now = time.perf_counter()
        with self.lock:
            refined_people = update_people_depth_from_boxes(self.latest_people, depth_m)
            if refined_people or not self.pending:
                self.latest_people = refined_people
            people = clone_people(self.latest_people)
            should_start = (
                not self.pending
                and now >= self.cooldown_until
                and rgb_frame is not None
                and depth_m is not None
                and (
                    self.frame_index == 1
                    or (self.frame_index - 1) % max(1, int(YOLO_EVERY_FRAMES)) == 0
                )
            )
            if should_start:
                self.pending = True
                rgb_for_worker = rgb_frame.copy()
                depth_for_worker = depth_m.copy()
            else:
                rgb_for_worker = None
                depth_for_worker = None

        if rgb_for_worker is not None and depth_for_worker is not None:
            self.worker = threading.Thread(
                target=self._detect_worker,
                args=(rgb_for_worker, depth_for_worker),
                daemon=True,
            )
            self.worker.start()

        return people

    def _detect_worker(self, rgb_frame, depth_m):
        started_at = time.perf_counter()
        try:
            people = self.detector.find_people(rgb_frame, depth_m)
        except Exception as exc:
            self.detector.error = f"YOLO worker failed: {exc}"
            people = []

        finished_at = time.perf_counter()
        with self.lock:
            elapsed_since_result = max(1e-3, finished_at - self.last_result_time)
            people, fast_object_seen = self._filter_fast_people(people, self.latest_people, elapsed_since_result)
            if fast_object_seen:
                self.cooldown_until = max(self.cooldown_until, finished_at + FAST_OBJECT_COOLDOWN_SECONDS)
                self.latest_people = []
            else:
                self.latest_people = people
            if finished_at - started_at > YOLO_SLOW_INFERENCE_SECONDS:
                self.cooldown_until = max(self.cooldown_until, finished_at + YOLO_SLOW_COOLDOWN_SECONDS)
            self.last_result_time = finished_at
            self.pending = False

    def _filter_fast_people(self, people, previous_people, elapsed_seconds):
        if not people or not previous_people:
            return people, False

        kept_people = []
        fast_object_seen = False
        for person in people:
            previous = min(
                previous_people,
                key=lambda candidate: math.hypot(person.x_norm - candidate.x_norm, person.y_norm - candidate.y_norm),
            )
            center_speed = math.hypot(person.x_norm - previous.x_norm, person.y_norm - previous.y_norm) / elapsed_seconds
            depth_speed = abs(person.distance_m - previous.distance_m) / elapsed_seconds
            if (
                center_speed > FAST_OBJECT_CENTER_SPEED_NORM_PER_SEC
                or depth_speed > FAST_OBJECT_DEPTH_SPEED_M_PER_SEC
            ):
                fast_object_seen = True
                continue

            kept_people.append(person)

        return kept_people, fast_object_seen

    def stop(self):
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=0.2)


def clip_bbox(x1, y1, x2, y2):
    x1 = int(np.clip(x1, 0, CAMERA_WIDTH - 1))
    y1 = int(np.clip(y1, 0, CAMERA_HEIGHT - 1))
    x2 = int(np.clip(x2, x1 + 1, CAMERA_WIDTH))
    y2 = int(np.clip(y2, y1 + 1, CAMERA_HEIGHT))
    return x1, y1, x2, y2


def person_from_yolo_box(bbox, confidence, depth_m):
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_h < CAMERA_HEIGHT * YOLO_MIN_BOX_HEIGHT_RATIO:
        return None
    if box_w < CAMERA_WIDTH * YOLO_MIN_BOX_WIDTH_RATIO:
        return None

    aspect_ratio = box_h / max(1.0, float(box_w))
    if aspect_ratio < YOLO_MIN_BOX_ASPECT_RATIO or aspect_ratio > YOLO_MAX_BOX_ASPECT_RATIO:
        return None

    distance_m = median_depth_in_yolo_box(depth_m, bbox)
    if distance_m is None:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return PersonState(
        present=True,
        x_norm=float(np.clip(cx / CAMERA_WIDTH, 0.0, 1.0)),
        y_norm=float(np.clip(cy / CAMERA_HEIGHT, 0.0, 1.0)),
        distance_m=distance_m,
        area=float(box_w * box_h),
        confidence=confidence,
        bbox=bbox,
    )


def median_depth_in_yolo_box(depth_m, bbox):
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    sample_w = max(12, int(box_w * YOLO_DEPTH_CENTER_WIDTH_RATIO))
    sample_h = max(12, int(box_h * YOLO_DEPTH_CENTER_HEIGHT_RATIO))

    sx1 = max(0, cx - sample_w // 2)
    sx2 = min(CAMERA_WIDTH, cx + sample_w // 2)
    sy1 = max(0, cy - sample_h // 2)
    sy2 = min(CAMERA_HEIGHT, cy + sample_h // 2)

    center_depth = depth_m[sy1:sy2, sx1:sx2]
    center_valid = center_depth[
        np.isfinite(center_depth)
        & (center_depth >= ACTIVE_NEAR_M)
        & (center_depth <= ACTIVE_FAR_M)
    ]
    if center_valid.size >= YOLO_MIN_DEPTH_PIXELS:
        return float(np.median(center_valid))

    box_depth = depth_m[y1:y2, x1:x2]
    box_valid = box_depth[
        np.isfinite(box_depth)
        & (box_depth >= ACTIVE_NEAR_M)
        & (box_depth <= ACTIVE_FAR_M)
    ]
    if box_valid.size >= YOLO_MIN_DEPTH_PIXELS:
        return float(np.median(box_valid))

    return None


def person_from_contour(contour, depth_m, mask):
    area = float(cv2.contourArea(contour))
    if area < MIN_PERSON_AREA:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w < MIN_PERSON_WIDTH_PX or h < MIN_PERSON_HEIGHT_PX:
        return None

    aspect_ratio = h / max(1.0, float(w))
    if aspect_ratio < MIN_PERSON_ASPECT_RATIO or aspect_ratio > MAX_PERSON_ASPECT_RATIO:
        return None

    fill_ratio = area / max(1.0, float(w * h))
    if fill_ratio > MAX_PERSON_FILL_RATIO:
        return None

    roi = depth_m[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w] > 0
    if not np.any(roi_mask):
        return None

    distance_m = float(np.median(roi[roi_mask]))

    # Nearby people should occupy more pixels. Farther people can be smaller.
    min_height_for_distance = int(np.interp(distance_m, [ACTIVE_NEAR_M, ACTIVE_FAR_M], [150, 70]))
    min_width_for_distance = int(np.interp(distance_m, [ACTIVE_NEAR_M, ACTIVE_FAR_M], [52, 24]))
    min_area_for_distance = float(np.interp(distance_m, [ACTIVE_NEAR_M, ACTIVE_FAR_M], [7000, 1800]))
    if h < min_height_for_distance or w < min_width_for_distance or area < min_area_for_distance:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    return PersonState(
        present=True,
        x_norm=float(np.clip(cx / CAMERA_WIDTH, 0.0, 1.0)),
        y_norm=float(np.clip(cy / CAMERA_HEIGHT, 0.0, 1.0)),
        distance_m=distance_m,
        area=area,
    )


def detect_person(depth_m):
    if depth_m is None:
        return PersonState()

    if depth_m.shape[1] != CAMERA_WIDTH or depth_m.shape[0] != CAMERA_HEIGHT:
        depth_m = cv2.resize(depth_m, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)

    depth_m = cv2.flip(depth_m, 1)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= ACTIVE_NEAR_M)
        & (depth_m <= ACTIVE_FAR_M)
    )
    if not np.any(valid):
        return PersonState()

    nearest_depth = float(np.percentile(depth_m[valid], 4))
    foreground = valid & (depth_m <= nearest_depth + FOREGROUND_BAND_M)

    mask = foreground.astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        person = person_from_contour(contour, depth_m, mask)
        if person is not None:
            candidates.append(person)

    if not candidates:
        return PersonState()

    return max(candidates, key=lambda candidate: candidate.area)


def detect_people(depth_m):
    person = detect_person(depth_m)
    return [person] if person.present else []


def distance_norm(distance_m):
    return float(np.clip((distance_m - ACTIVE_NEAR_M) / (ACTIVE_FAR_M - ACTIVE_NEAR_M), 0.0, 1.0))


def mix_color(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def get_depth_zone(distance_m):
    for zone in DEPTH_ZONES:
        if zone["min_m"] <= distance_m < zone["max_m"]:
            return zone
    if math.isclose(distance_m, ACTIVE_FAR_M):
        return DEPTH_ZONES[-1]
    return None


def zone_color_for_distance(distance_m):
    zone = get_depth_zone(distance_m)
    if zone is None:
        return (90, 225, 255)
    return zone["color"]


def floor_width_at(z_norm):
    return 575.0 * ((1.0 - z_norm) ** 0.72) + 108.0 * z_norm


def floor_y_at(z_norm):
    return FLOOR_HORIZON_Y + (FLOOR_NEAR_Y - FLOOR_HORIZON_Y) * ((1.0 - z_norm) ** 0.58)


def project_floor(x_m, distance_m):
    z_norm = distance_norm(distance_m)
    half_width = floor_width_at(z_norm)
    y = floor_y_at(z_norm)
    x = SCREEN_WIDTH / 2 + (x_m / FLOOR_HALF_WIDTH_M) * half_width
    return int(x), int(y)


def make_background(t):
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    y = np.linspace(0.0, 1.0, SCREEN_HEIGHT, dtype=np.float32)[:, None]
    frame[:, :, 0] = np.clip(26 + 25 * y, 0, 255)
    frame[:, :, 1] = np.clip(18 + 18 * y, 0, 255)
    frame[:, :, 2] = np.clip(36 + 12 * np.sin(t * 0.5) + 18 * (1.0 - y), 0, 255)
    return frame


def draw_text(frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(frame, text, y, scale=0.82, color=(255, 255, 255), thickness=2):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(12, (SCREEN_WIDTH - text_w) // 2)
    draw_text(frame, text, (x, y + text_h // 2), scale, color, thickness)


def draw_floor(frame, t, active_zone=None):
    cv2.rectangle(frame, (0, FLOOR_HORIZON_Y - 80), (SCREEN_WIDTH, SCREEN_HEIGHT), (16, 18, 34), -1)

    floor_poly = np.array(
        [
            [SCREEN_WIDTH // 2 - 118, FLOOR_HORIZON_Y],
            [SCREEN_WIDTH // 2 + 118, FLOOR_HORIZON_Y],
            [SCREEN_WIDTH // 2 + 590, FLOOR_NEAR_Y],
            [SCREEN_WIDTH // 2 - 590, FLOOR_NEAR_Y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, floor_poly, (24, 31, 50))

    zone_overlay = np.zeros_like(frame)
    for zone in DEPTH_ZONES:
        near_left = project_floor(-FLOOR_HALF_WIDTH_M, zone["min_m"])
        near_right = project_floor(FLOOR_HALF_WIDTH_M, zone["min_m"])
        far_right = project_floor(FLOOR_HALF_WIDTH_M, zone["max_m"])
        far_left = project_floor(-FLOOR_HALF_WIDTH_M, zone["max_m"])
        zone_poly = np.array([near_left, near_right, far_right, far_left], dtype=np.int32)

        is_active = active_zone is not None and zone["name"] == active_zone["name"]
        cv2.fillConvexPoly(zone_overlay, zone_poly, zone["color"])
        cv2.polylines(
            frame,
            [zone_poly],
            True,
            mix_color(zone["color"], (255, 255, 255), 0.22 if is_active else 0.0),
            3 if is_active else 1,
            cv2.LINE_AA,
        )

        mid_m = (zone["min_m"] + zone["max_m"]) / 2.0
        label_x, label_y = project_floor(0.0, mid_m)
        label_color = mix_color(zone["color"], (255, 255, 255), 0.52 if is_active else 0.25)
        draw_text(frame, zone["name"], (label_x - 34, label_y + 7), 0.58, label_color, 1)

    cv2.addWeighted(zone_overlay, 0.17, frame, 1.0, 0, dst=frame)

    if active_zone is not None:
        zone_pulse = 0.45 + 0.22 * math.sin(t * 4.0)
        glow = np.zeros_like(frame)
        near_left = project_floor(-FLOOR_HALF_WIDTH_M, active_zone["min_m"])
        near_right = project_floor(FLOOR_HALF_WIDTH_M, active_zone["min_m"])
        far_right = project_floor(FLOOR_HALF_WIDTH_M, active_zone["max_m"])
        far_left = project_floor(-FLOOR_HALF_WIDTH_M, active_zone["max_m"])
        zone_poly = np.array([near_left, near_right, far_right, far_left], dtype=np.int32)
        cv2.fillConvexPoly(glow, zone_poly, active_zone["color"])
        glow = cv2.GaussianBlur(glow, (0, 0), 18)
        cv2.addWeighted(glow, zone_pulse, frame, 1.0, 0, dst=frame)

    for distance_m in np.arange(ACTIVE_NEAR_M, ACTIVE_FAR_M + 0.001, 0.5):
        z_norm = distance_norm(distance_m)
        half_width = floor_width_at(z_norm)
        y = int(floor_y_at(z_norm))
        color = (55, 92, 120) if abs(distance_m - round(distance_m)) > 0.01 else (75, 135, 165)
        cv2.line(
            frame,
            (int(SCREEN_WIDTH / 2 - half_width), y),
            (int(SCREEN_WIDTH / 2 + half_width), y),
            color,
            1,
            cv2.LINE_AA,
        )
        if abs(distance_m - round(distance_m)) < 0.01:
            draw_text(frame, f"{distance_m:.0f} m", (int(SCREEN_WIDTH / 2 + half_width + 14), y + 5), 0.46, (180, 220, 235), 1)

    for x_m in np.linspace(-FLOOR_HALF_WIDTH_M, FLOOR_HALF_WIDTH_M, 9):
        near = project_floor(float(x_m), ACTIVE_NEAR_M)
        far = project_floor(float(x_m), ACTIVE_FAR_M)
        cv2.line(frame, near, far, (45, 76, 105), 1, cv2.LINE_AA)

    glow = np.zeros_like(frame)
    cv2.ellipse(glow, (SCREEN_WIDTH // 2, FLOOR_NEAR_Y + 28), (560, 92), 0, 180, 360, (80, 200, 255), 4, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 14)
    cv2.addWeighted(glow, 0.75 + 0.15 * math.sin(t * 2.2), frame, 1.0, 0, dst=frame)


def create_zone_glow_frames():
    zone_glows = {}
    for zone in DEPTH_ZONES:
        glow = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        near_left = project_floor(-FLOOR_HALF_WIDTH_M, zone["min_m"])
        near_right = project_floor(FLOOR_HALF_WIDTH_M, zone["min_m"])
        far_right = project_floor(FLOOR_HALF_WIDTH_M, zone["max_m"])
        far_left = project_floor(-FLOOR_HALF_WIDTH_M, zone["max_m"])
        zone_poly = np.array([near_left, near_right, far_right, far_left], dtype=np.int32)
        cv2.fillConvexPoly(glow, zone_poly, zone["color"])
        zone_glows[zone["name"]] = cv2.GaussianBlur(glow, (0, 0), 18)
    return zone_glows


def draw_active_zone_glow(frame, active_zone, t, zone_glows):
    if active_zone is None:
        return

    glow = zone_glows.get(active_zone["name"])
    if glow is None:
        return

    zone_pulse = 0.42 + 0.2 * math.sin(t * 4.0)
    cv2.addWeighted(glow, zone_pulse, frame, 1.0, 0, dst=frame)


def update_avatar_trails(trail_layer, people):
    trail_layer = cv2.convertScaleAbs(trail_layer, alpha=AVATAR_TRAIL_DECAY, beta=0)
    trail_layer = cv2.GaussianBlur(trail_layer, (0, 0), 1.2)

    for person in people:
        x_m = (person.x_norm - 0.5) * 2.0 * FLOOR_HALF_WIDTH_M
        px, py = project_floor(x_m, person.distance_m)
        scale_x = trail_layer.shape[1] / SCREEN_WIDTH
        scale_y = trail_layer.shape[0] / SCREEN_HEIGHT
        px = int(px * scale_x)
        py = int(py * scale_y)
        z_norm = distance_norm(person.distance_m)
        radius = max(4, int((24 + (1.0 - z_norm) * 38) * min(scale_x, scale_y)))
        color = zone_color_for_distance(person.distance_m)
        cv2.circle(trail_layer, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(trail_layer, (px, py), max(2, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)

    return trail_layer


def draw_avatar_trail(frame, trail_layer):
    blur_sigma = max(2.0, 10.0 * min(trail_layer.shape[1] / SCREEN_WIDTH, trail_layer.shape[0] / SCREEN_HEIGHT))
    blurred = cv2.GaussianBlur(trail_layer, (0, 0), blur_sigma)
    if blurred.shape[1] != SCREEN_WIDTH or blurred.shape[0] != SCREEN_HEIGHT:
        blurred = cv2.resize(blurred, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(blurred, AVATAR_TRAIL_WEIGHT, frame, 1.0, 0, dst=frame)


def create_avatar_trail_layer():
    trail_w = max(1, int(SCREEN_WIDTH * AVATAR_TRAIL_SCALE))
    trail_h = max(1, int(SCREEN_HEIGHT * AVATAR_TRAIL_SCALE))
    return np.zeros((trail_h, trail_w, 3), dtype=np.uint8)


def draw_camera(frame):
    cx, cy = CAMERA_SCREEN_POS
    cv2.line(frame, (cx, cy - 18), project_floor(-FLOOR_HALF_WIDTH_M, ACTIVE_FAR_M), (50, 145, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 18), project_floor(FLOOR_HALF_WIDTH_M, ACTIVE_FAR_M), (50, 145, 190), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - 58, cy - 34), (cx + 58, cy + 20), (8, 10, 18), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - 58, cy - 34), (cx + 58, cy + 20), (110, 210, 240), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - 8), 16, (25, 45, 72), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - 8), 8, (90, 225, 255), -1, cv2.LINE_AA)
    draw_text(frame, "HP60", (cx - 27, cy + 49), 0.54, (210, 235, 245), 1)


def create_base_scene():
    frame = make_background(0.0)
    draw_floor(frame, 0.0)
    draw_camera(frame)
    return frame


def add_avatar_glow(frame, px, py, body_h, body_w, aura_size, scale, zone_color):
    margin = int(92 + aura_size)
    x0 = max(0, px - body_w - margin)
    x1 = min(SCREEN_WIDTH, px + body_w + margin)
    y0 = max(0, py - body_h - margin)
    y1 = min(SCREEN_HEIGHT, py + margin)
    if x0 >= x1 or y0 >= y1:
        return

    glow = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    local_px = px - x0
    local_py = py - y0
    cv2.ellipse(
        glow,
        (local_px, local_py - body_h // 2),
        (body_w + aura_size, body_h // 2 + aura_size),
        0,
        0,
        360,
        zone_color,
        5,
        cv2.LINE_AA,
    )
    cv2.circle(glow, (local_px, local_py), int(34 + scale * 46), zone_color, 3, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 13)
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(glow, 1.05, roi, 1.0, 0, dst=roi)


def draw_person(frame, person, t):
    x_m = (person.x_norm - 0.5) * 2.0 * FLOOR_HALF_WIDTH_M
    px, py = project_floor(x_m, person.distance_m)
    z_norm = distance_norm(person.distance_m)
    scale = 1.0 - z_norm
    zone_color = zone_color_for_distance(person.distance_m)
    body_color = mix_color(zone_color, (255, 255, 255), 0.08)
    limb_color = mix_color(zone_color, (255, 255, 255), 0.28)
    head_color = mix_color(zone_color, (255, 255, 255), 0.62)
    body_h = int(72 + scale * 116)
    body_w = int(34 + scale * 46)
    head_r = int(13 + scale * 16)
    body_top = py - body_h
    body_bottom = py - 18

    shadow_w = int(body_w * 1.95)
    shadow_h = int(12 + scale * 18)
    cv2.ellipse(frame, (px, py + 8), (shadow_w, shadow_h), 0, 0, 360, (4, 6, 12), -1, cv2.LINE_AA)

    aura_size = int(26 + scale * 34 + 8 * math.sin(t * 5.0))
    add_avatar_glow(frame, px, py, body_h, body_w, aura_size, scale, zone_color)

    sway = int(math.sin(t * 3.2) * 3)
    cv2.line(frame, (px - body_w // 2, body_top + 45), (px - body_w - 18, body_bottom - 35 + sway), limb_color, 8, cv2.LINE_AA)
    cv2.line(frame, (px + body_w // 2, body_top + 45), (px + body_w + 18, body_bottom - 35 - sway), limb_color, 8, cv2.LINE_AA)
    cv2.line(frame, (px - body_w // 4, body_bottom), (px - body_w // 2, py), body_color, 9, cv2.LINE_AA)
    cv2.line(frame, (px + body_w // 4, body_bottom), (px + body_w // 2, py), body_color, 9, cv2.LINE_AA)
    cv2.ellipse(frame, (px, body_top + 70), (body_w, body_h // 3), 0, 0, 360, body_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, body_top + 25), head_r, head_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, body_top + 25), head_r + 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), int(18 + scale * 22), mix_color(zone_color, (255, 255, 255), 0.35), 2, cv2.LINE_AA)

    cv2.line(frame, CAMERA_SCREEN_POS, (px, py), mix_color(zone_color, (255, 255, 255), 0.34), 3, cv2.LINE_AA)
    mid_x = int((CAMERA_SCREEN_POS[0] + px) / 2)
    mid_y = int((CAMERA_SCREEN_POS[1] + py) / 2)
    label = f"#{person.track_id}  {person.distance_m:.2f} m" if person.track_id else f"{person.distance_m:.2f} m"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.86, 2)
    cv2.rectangle(frame, (mid_x - text_w // 2 - 16, mid_y - text_h - 18), (mid_x + text_w // 2 + 16, mid_y + 12), (0, 0, 0), -1, cv2.LINE_AA)
    draw_text(frame, label, (mid_x - text_w // 2, mid_y), 0.86, (130, 255, 180), 2)

    return px, py


def draw_distance_panel(frame, people):
    panel_x = 36
    panel_y = 86
    panel_w = 336
    panel_h = 252
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (5, 8, 18), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (75, 165, 205), 2)

    draw_text(frame, "Tracked Humans", (panel_x + 22, panel_y + 42), 0.72, (90, 225, 255), 2)
    if people:
        closest = people[0]
        draw_text(frame, f"{len(people)} visible", (panel_x + 22, panel_y + 86), 0.72, (135, 255, 170), 2)
        draw_text(frame, f"closest {closest.distance_m:.2f} m", (panel_x + 22, panel_y + 124), 0.64, (225, 235, 240), 1)

        for index, person in enumerate(people[:4]):
            zone = get_depth_zone(person.distance_m)
            zone_color = zone["color"] if zone is not None else (90, 225, 255)
            row_y = panel_y + 158 + index * 22
            cv2.circle(frame, (panel_x + 32, row_y - 5), 6, zone_color, -1, cv2.LINE_AA)
            zone_name = zone["name"] if zone is not None else "OUT"
            draw_text(
                frame,
                f"#{person.track_id}  {person.distance_m:.2f} m  {zone_name}",
                (panel_x + 48, row_y),
                0.5,
                (225, 235, 240),
                1,
            )
    else:
        draw_text(frame, "--.-- m", (panel_x + 22, panel_y + 102), 1.35, (180, 180, 180), 3)
        draw_text(frame, "waiting for humans", (panel_x + 24, panel_y + 146), 0.58, (225, 235, 240), 1)


def draw_range_bar(frame, people):
    x0 = SCREEN_WIDTH - 348
    y0 = 100
    w = 280
    h = 18
    cv2.rectangle(frame, (x0 - 18, y0 - 44), (x0 + w + 18, y0 + 84), (5, 8, 18), -1)
    cv2.rectangle(frame, (x0 - 18, y0 - 44), (x0 + w + 18, y0 + 84), (75, 165, 205), 2)
    draw_text(frame, "Play range", (x0, y0 - 14), 0.62, (90, 225, 255), 2)

    for zone in DEPTH_ZONES:
        sx = int(x0 + distance_norm(zone["min_m"]) * w)
        ex = int(x0 + distance_norm(zone["max_m"]) * w)
        cv2.rectangle(frame, (sx, y0), (ex, y0 + h), zone["color"], -1)
        draw_text(frame, zone["name"], (sx + 5, y0 + 43), 0.42, (225, 235, 240), 1)

    for zone in DEPTH_ZONES[1:]:
        bx = int(x0 + distance_norm(zone["min_m"]) * w)
        cv2.line(frame, (bx, y0 - 5), (bx, y0 + h + 5), (255, 255, 255), 1, cv2.LINE_AA)

    draw_text(frame, f"{ACTIVE_NEAR_M:.1f} m", (x0, y0 + 68), 0.5, (225, 235, 240), 1)
    draw_text(frame, f"{ACTIVE_FAR_M:.1f} m", (x0 + w - 54, y0 + 68), 0.5, (225, 235, 240), 1)

    for person in people:
        marker_x = int(x0 + distance_norm(person.distance_m) * w)
        color = zone_color_for_distance(person.distance_m)
        cv2.rectangle(frame, (marker_x - 6, y0 - 8), (marker_x + 6, y0 + h + 8), (255, 255, 255), -1)
        cv2.rectangle(frame, (marker_x - 4, y0 - 6), (marker_x + 4, y0 + h + 6), color, -1)


def draw_header(frame, fps_value):
    cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, 62), (0, 0, 0), -1)
    draw_text(frame, "Depth Floor Distance Simulator", (30, 40), 0.86, (90, 225, 255), 2)
    if SHOW_FPS_COUNTER:
        cap_text = "uncapped" if MAX_PROGRAM_FPS <= 0 else f"cap {MAX_PROGRAM_FPS:g}"
        draw_text(frame, f"FPS {fps_value:4.1f}  {cap_text}", (520, 38), 0.58, (135, 255, 170), 1)
    draw_text(frame, "D debug   F fullscreen   R reset   Q quit", (SCREEN_WIDTH - 446, 38), 0.58, (225, 235, 240), 1)


def draw_footer_hint(frame, people):
    if people:
        closest = people[0]
        zone = get_depth_zone(closest.distance_m)
        if zone is not None:
            draw_centered_text(frame, f"Tracking {len(people)} humans - closest is in the {zone['label']}", SCREEN_HEIGHT - 36, 0.64, (255, 255, 255), 1)
        else:
            draw_centered_text(frame, "Move closer or farther and watch each human marker move on the floor", SCREEN_HEIGHT - 36, 0.64, (255, 255, 255), 1)
    else:
        draw_centered_text(frame, "Stand between 0.5 m and 3.5 m from the camera", SCREEN_HEIGHT - 36, 0.64, (255, 255, 255), 1)


def main():
    camera = HP60SDKCamera(sdk_root=SDK_ROOT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS)
    camera.start()
    yolo_detector = AsyncYOLOPersonDetector() if YOLO_ASYNC_DETECTION else YOLOPersonDetector()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)

    smoother = MultiPersonSmoother()
    fullscreen = False
    debug_enabled = RGB_DEPTH_DEBUG
    debug_windows_open = False
    base_scene = create_base_scene()
    zone_glows = create_zone_glow_frames()
    avatar_trail = create_avatar_trail_layer()
    fps_meter = FPSMeter()

    try:
        while True:
            loop_start_time = time.perf_counter()
            fps_value = fps_meter.update()
            now = time.time()
            rgb_frame, raw_depth = camera.get_latest_frames()
            depth_m = depth_to_meters(raw_depth)
            if yolo_detector.ready:
                measured_people = yolo_detector.detect(rgb_frame, depth_m)
            elif USE_YOLO_PERSON_DETECTION:
                measured_people = []
            else:
                measured_people = detect_people(depth_m)
            people = smoother.update(measured_people)
            active_zones = {
                zone["name"]: zone
                for zone in (get_depth_zone(person.distance_m) for person in people)
                if zone is not None
            }

            frame = base_scene.copy()
            for active_zone in active_zones.values():
                draw_active_zone_glow(frame, active_zone, now, zone_glows)
            avatar_trail = update_avatar_trails(avatar_trail, people)
            draw_avatar_trail(frame, avatar_trail)
            for person in reversed(people):
                draw_person(frame, person, now)
            draw_distance_panel(frame, people)
            draw_header(frame, fps_value)
            draw_footer_hint(frame, people)
            if USE_YOLO_PERSON_DETECTION and not yolo_detector.ready:
                draw_centered_text(frame, yolo_detector.error, SCREEN_HEIGHT - 92, 0.62, (100, 200, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)
            if debug_enabled:
                draw_debug_windows(rgb_frame, depth_m, people)
                debug_windows_open = True
            elif debug_windows_open:
                close_debug_windows()
                debug_windows_open = False

            limit_program_fps(loop_start_time)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("d"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    close_debug_windows()
                    debug_windows_open = False
            if key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            if key == ord("r"):
                smoother = MultiPersonSmoother()
                avatar_trail[:] = 0
    finally:
        if hasattr(yolo_detector, "stop"):
            yolo_detector.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
