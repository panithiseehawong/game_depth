from dataclasses import dataclass
from pathlib import Path
import random
import time

import cv2
import numpy as np

from hp60_sdk import HP60SDKCamera


cv2.setUseOptimized(True)

PROJECT_DIR = Path(__file__).resolve().parent
SDK_ROOT = PROJECT_DIR / "EaiCameraSdk_v1.2.28.20241015"
HIGH_SCORE_FILE = PROJECT_DIR / "sequence_memory_best_level.txt"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WINDOW_NAME = "Sequence Memory Depth Game"
RGB_DEBUG_WINDOW = "Sequence RGB Debug"
DEPTH_DEBUG_WINDOW = "Sequence Depth Debug"

RGB_DEPTH_DEBUG = False ####################################################3
SHOW_FPS_COUNTER = True
SHOW_BIG_COUNTDOWN = False
MAX_PROGRAM_FPS = 60.0
FPS_COUNTER_SMOOTHING = 0.9

USE_YOLO_PERSON_DETECTION = True
YOLO_MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_PERSON_CONFIDENCE = 0.55
YOLO_EVERY_FRAMES = 5 ########################################################SSSSSS
YOLO_IMAGE_SIZE = 256 ###############################################################################
YOLO_DEVICE = "cpu"
YOLO_USE_HALF_ON_CUDA = False
YOLO_MIN_BOX_HEIGHT_RATIO = 0.28
YOLO_MIN_BOX_WIDTH_RATIO = 0.06
YOLO_MIN_BOX_ASPECT_RATIO = 1.0
YOLO_MAX_BOX_ASPECT_RATIO = 6.2
YOLO_DEPTH_CENTER_WIDTH_RATIO = 0.34
YOLO_DEPTH_CENTER_HEIGHT_RATIO = 0.28
YOLO_MIN_DEPTH_PIXELS = 25

SQUARE_SIZE_DISTANSE = 2 ############################################################################
SQUARE_SIZE_PROFILES = {
    1: {
        "name": "small room",
        "x_min_norm": 0.30,
        "x_max_norm": 0.70,
        "near_m": 0.7,
        "near_row_max_m": 1.15,
        "middle_row_max_m": 1.65,
        "far_m": 2.1,
    },
    2: {
        "name": "normal",
        "x_min_norm": 0.0,
        "x_max_norm": 1.0,
        "near_m": 0.5,
        "near_row_max_m": 1.4,
        "middle_row_max_m": 2.4,
        "far_m": 3.5,
    },
}
SQUARE_PROFILE = SQUARE_SIZE_PROFILES.get(SQUARE_SIZE_DISTANSE, SQUARE_SIZE_PROFILES[2])
ACTIVE_NEAR_M = SQUARE_PROFILE["near_m"]
ACTIVE_FAR_M = SQUARE_PROFILE["far_m"]
NEAR_ROW_MAX_M = SQUARE_PROFILE["near_row_max_m"]
MIDDLE_ROW_MAX_M = SQUARE_PROFILE["middle_row_max_m"]
COLUMN_X_MIN_NORM = SQUARE_PROFILE["x_min_norm"]
COLUMN_X_MAX_NORM = SQUARE_PROFILE["x_max_norm"]

START_SEQUENCE_LENGTH = 4
START_CELL = 4
AUTO_START_WITH_CENTER = True
START_HOLD_SECONDS = 5.0
FLASH_SECONDS = 0.52
GAP_SECONDS = 0.18
HOLD_TO_SELECT_SECONDS = 1.0
NEXT_LEVEL_HOLD_SECONDS = 3.0
DIFFICULTY_SELECT_HOLD_SECONDS = 3.0
INPUT_FEEDBACK_SECONDS = 0.28
PLAYER_MARKER_RADIUS = 28
BACKGROUND_CACHE = None

NORMAL_DIFFICULTY_CELL = 3
HARD_DIFFICULTY_CELL = 5
DEFAULT_DIFFICULTY = "normal"
DIFFICULTY_SETTINGS = {
    "normal": {
        "label": "Normal",
        "color": (95, 235, 145),
        "start_length": 4,
        "level_increment": 1,
        "flash_seconds": 0.52,
        "gap_seconds": 0.18,
    },
    "hard": {
        "label": "Hard",
        "color": (70, 115, 255),
        "start_length": 4,
        "level_increment": 2,
        "flash_seconds": 0.34,
        "gap_seconds": 0.10,
    },
}

GRID_SIZE = 540
GRID_GAP = 14
CELL_SIZE = (GRID_SIZE - GRID_GAP * 2) // 3
GRID_X = (SCREEN_WIDTH - GRID_SIZE) // 2
GRID_Y = 112

ROW_INFO = [
    ("NEAR", (45, 125, 255)),
    ("MIDDLE", (255, 150, 70)),
    ("FAR", (255, 110, 215)),
]


@dataclass
class PersonState:
    present: bool = False
    x_norm: float = 0.5
    y_norm: float = 0.5
    distance_m: float = 0.0
    area: float = 0.0
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


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
            if self.stable_seen_frames < 2 and not self.state.present:
                return self.state
            if not self.state.present:
                self.state = measured
            else:
                alpha = 0.24
                self.state.x_norm = self.state.x_norm * (1.0 - alpha) + measured.x_norm * alpha
                self.state.y_norm = self.state.y_norm * (1.0 - alpha) + measured.y_norm * alpha
                self.state.distance_m = self.state.distance_m * (1.0 - alpha) + measured.distance_m * alpha
                self.state.area = self.state.area * (1.0 - alpha) + measured.area * alpha
                self.state.confidence = measured.confidence
                self.state.bbox = measured.bbox
                self.state.present = True
        else:
            self.stable_seen_frames = 0

        if now - self.last_seen_time > 0.35:
            self.state.present = False
        return self.state


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


class SequenceGame:
    def __init__(self):
        self.best_level = read_best_level()
        self.state = "waiting"
        self.level = 1
        self.sequence = []
        self.input_index = 0
        self.show_start_time = 0.0
        self.hold_cell = None
        self.hold_start_time = 0.0
        self.hold_progress = 0.0
        self.feedback_cell = None
        self.feedback_color = (255, 255, 255)
        self.feedback_until = 0.0
        self.start_hold_start_time = 0.0
        self.start_hold_progress = 0.0
        self.next_level_hold_start_time = 0.0
        self.next_level_hold_progress = 0.0
        self.center_locked_until_exit = False
        self.released_after_submit = True
        self.last_submitted_cell = None
        self.difficulty = DEFAULT_DIFFICULTY
        self.difficulty_hold_cell = None
        self.difficulty_hold_start_time = 0.0
        self.difficulty_hold_progress = 0.0

    def difficulty_settings(self):
        return DIFFICULTY_SETTINGS[self.difficulty]

    def start(self, now):
        self.level = 1
        self.sequence = generate_sequence_path(self.difficulty_settings()["start_length"])
        self.best_level = max(self.best_level, 1)
        self.clear_start_hold()
        self.clear_difficulty_hold()
        self.start_showing(now)

    def update_difficulty_control(self, now, selected_cell):
        if self.state not in ("waiting", "game_over"):
            return

        difficulty_by_cell = {
            NORMAL_DIFFICULTY_CELL: "normal",
            HARD_DIFFICULTY_CELL: "hard",
        }
        if selected_cell not in difficulty_by_cell:
            self.clear_difficulty_hold()
            return

        if selected_cell != self.difficulty_hold_cell:
            self.difficulty_hold_cell = selected_cell
            self.difficulty_hold_start_time = now
            self.difficulty_hold_progress = 0.0
            return

        self.difficulty_hold_progress = min(1.0, (now - self.difficulty_hold_start_time) / DIFFICULTY_SELECT_HOLD_SECONDS)
        if self.difficulty_hold_progress >= 1.0:
            self.difficulty = difficulty_by_cell[selected_cell]
            self.clear_difficulty_hold()

    def update_start_control(self, now, selected_cell):
        if not AUTO_START_WITH_CENTER or self.state not in ("waiting", "game_over"):
            return

        if selected_cell != START_CELL:
            self.clear_start_hold()
            return

        if self.start_hold_start_time <= 0:
            self.start_hold_start_time = now
            self.start_hold_progress = 0.0
            return

        self.start_hold_progress = min(1.0, (now - self.start_hold_start_time) / START_HOLD_SECONDS)
        if self.start_hold_progress >= 1.0:
            self.start(now)

    def start_showing(self, now):
        self.state = "showing"
        self.show_start_time = now
        self.input_index = 0
        self.center_locked_until_exit = True
        self.released_after_submit = True
        self.last_submitted_cell = None
        self.clear_hold()
        self.clear_next_level_hold()

    def current_flash_cell(self, now):
        if self.state != "showing":
            return None

        settings = self.difficulty_settings()
        flash_seconds = settings["flash_seconds"]
        interval = flash_seconds + settings["gap_seconds"]
        elapsed = now - self.show_start_time
        step_index = int(elapsed / interval)
        if step_index >= len(self.sequence):
            self.state = "input"
            self.clear_hold()
            return None

        if elapsed % interval < flash_seconds:
            return self.sequence[step_index]
        return None

    def update(self, now, selected_cell):
        if self.state == "level_clear":
            self.update_next_level_control(now, selected_cell)
            return

        if self.state != "input":
            return

        if selected_cell is None:
            self.released_after_submit = True
            self.clear_hold()
            return

        if self.center_locked_until_exit:
            if selected_cell == START_CELL:
                self.clear_hold()
                return
            self.center_locked_until_exit = False

        if not self.released_after_submit:
            if selected_cell == self.last_submitted_cell:
                self.clear_hold()
                return
            self.released_after_submit = True

        if selected_cell != self.hold_cell:
            self.hold_cell = selected_cell
            self.hold_start_time = now
            self.hold_progress = 0.0
            return

        self.hold_progress = min(1.0, (now - self.hold_start_time) / HOLD_TO_SELECT_SECONDS)
        if self.hold_progress >= 1.0:
            self.submit_cell(now, selected_cell)

    def submit_cell(self, now, cell):
        expected = self.sequence[self.input_index]
        if cell != expected:
            self.feedback_cell = cell
            self.feedback_color = (40, 60, 255)
            self.feedback_until = now + INPUT_FEEDBACK_SECONDS
            self.state = "game_over"
            self.clear_hold()
            return

        self.feedback_cell = cell
        self.feedback_color = (90, 255, 135)
        self.feedback_until = now + INPUT_FEEDBACK_SECONDS
        self.input_index += 1
        self.last_submitted_cell = cell
        self.released_after_submit = False
        self.clear_hold()

        if self.input_index >= len(self.sequence):
            self.best_level = max(self.best_level, self.level)
            write_best_level(self.best_level)
            self.state = "level_clear"
            self.clear_next_level_hold()

    def clear_hold(self):
        self.hold_cell = None
        self.hold_start_time = 0.0
        self.hold_progress = 0.0

    def clear_start_hold(self):
        self.start_hold_start_time = 0.0
        self.start_hold_progress = 0.0

    def update_next_level_control(self, now, selected_cell):
        if selected_cell != START_CELL:
            self.clear_next_level_hold()
            return

        if self.next_level_hold_start_time <= 0:
            self.next_level_hold_start_time = now
            self.next_level_hold_progress = 0.0
            return

        self.next_level_hold_progress = min(1.0, (now - self.next_level_hold_start_time) / NEXT_LEVEL_HOLD_SECONDS)
        if self.next_level_hold_progress >= 1.0:
            self.level += 1
            previous_cell = self.sequence[-1] if self.sequence else START_CELL
            for _ in range(self.difficulty_settings()["level_increment"]):
                previous_cell = random_next_cell(previous_cell)
                self.sequence.append(previous_cell)
            self.start_showing(now)

    def clear_next_level_hold(self):
        self.next_level_hold_start_time = 0.0
        self.next_level_hold_progress = 0.0

    def clear_difficulty_hold(self):
        self.difficulty_hold_cell = None
        self.difficulty_hold_start_time = 0.0
        self.difficulty_hold_progress = 0.0


class YOLOPersonDetector:
    def __init__(self):
        self.model = None
        self.error = ""
        self.enabled = USE_YOLO_PERSON_DETECTION
        self.frame_index = 0
        self.cached_person = PersonState()
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

            if YOLO_DEVICE == "auto":
                import torch

                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
            return PersonState()

        self.frame_index += 1
        interval = max(1, int(YOLO_EVERY_FRAMES))
        should_run_yolo = self.frame_index == 1 or (self.frame_index - 1) % interval == 0
        if not should_run_yolo:
            return self.update_cached_depth(depth_m)

        rgb = prepare_rgb_for_detection(rgb_frame)
        depth = prepare_depth_for_detection(depth_m)
        if rgb is None or depth is None:
            self.cached_person = PersonState()
            return self.cached_person

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
            self.cached_person = PersonState()
            return self.cached_person

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self.cached_person = PersonState()
            return self.cached_person

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

        if not candidates:
            self.cached_person = PersonState()
            return self.cached_person

        self.cached_person = max(candidates, key=lambda candidate: candidate.confidence * max(1.0, candidate.area))
        return self.cached_person

    def update_cached_depth(self, depth_m):
        if not self.cached_person.present or self.cached_person.bbox is None:
            return PersonState()

        depth = prepare_depth_for_detection(depth_m)
        if depth is None:
            self.cached_person = PersonState()
            return self.cached_person

        distance_m = median_depth_in_yolo_box(depth, self.cached_person.bbox)
        if distance_m is None:
            self.cached_person = PersonState()
            return self.cached_person

        self.cached_person.distance_m = distance_m
        return self.cached_person


def read_best_level():
    try:
        return int(HIGH_SCORE_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 1


def write_best_level(level):
    HIGH_SCORE_FILE.write_text(str(int(level)), encoding="utf-8")


def adjacent_cells(cell):
    row = cell // 3
    col = cell % 3
    neighbors = []
    for row_step, col_step in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        next_row = row + row_step
        next_col = col + col_step
        if 0 <= next_row < 3 and 0 <= next_col < 3:
            neighbors.append(next_row * 3 + next_col)
    return neighbors


def random_next_cell(previous_cell):
    return random.choice(adjacent_cells(previous_cell))


def generate_sequence_path(length):
    sequence = []
    previous_cell = START_CELL
    for _ in range(length):
        next_cell = random_next_cell(previous_cell)
        sequence.append(next_cell)
        previous_cell = next_cell
    return sequence


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


def selected_cell_from_person(person):
    if not person.present:
        return None

    x_norm = normalized_grid_x(person.x_norm)
    col = int(np.clip(x_norm * 3.0, 0, 2))
    if person.distance_m < NEAR_ROW_MAX_M:
        row = 0
    elif person.distance_m < MIDDLE_ROW_MAX_M:
        row = 1
    else:
        row = 2
    return row * 3 + col


def normalized_grid_x(x_norm):
    width = max(0.01, COLUMN_X_MAX_NORM - COLUMN_X_MIN_NORM)
    return float(np.clip((x_norm - COLUMN_X_MIN_NORM) / width, 0.0, 1.0))


def person_grid_position(person):
    if not person.present:
        return None

    x = GRID_X + normalized_grid_x(person.x_norm) * GRID_SIZE
    depth_t = float(np.clip((person.distance_m - ACTIVE_NEAR_M) / (ACTIVE_FAR_M - ACTIVE_NEAR_M), 0.0, 1.0))
    y = GRID_Y + depth_t * GRID_SIZE
    return int(x), int(y)


def cell_rect(cell):
    row = cell // 3
    col = cell % 3
    x = GRID_X + col * (CELL_SIZE + GRID_GAP)
    y = GRID_Y + row * (CELL_SIZE + GRID_GAP)
    return x, y, x + CELL_SIZE, y + CELL_SIZE


def cell_center(cell):
    x1, y1, x2, y2 = cell_rect(cell)
    return (x1 + x2) // 2, (y1 + y2) // 2


def make_background(t):
    global BACKGROUND_CACHE
    if BACKGROUND_CACHE is not None:
        return BACKGROUND_CACHE.copy()

    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    yy = np.linspace(0.0, 1.0, SCREEN_HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, SCREEN_WIDTH, dtype=np.float32)[None, :]
    wave = 0.5 + 0.5 * np.sin((xx * 1.4 + yy * 0.7) * np.pi)
    frame[:, :, 0] = np.clip(22 + 26 * yy + 10 * wave, 0, 255)
    frame[:, :, 1] = np.clip(18 + 18 * (1.0 - yy) + 12 * xx, 0, 255)
    frame[:, :, 2] = np.clip(24 + 18 * (1.0 - yy) + 8 * (1.0 - xx), 0, 255)

    for x in range(-SCREEN_HEIGHT, SCREEN_WIDTH, 72):
        cv2.line(frame, (x, SCREEN_HEIGHT), (x + SCREEN_HEIGHT, 0), (34, 52, 60), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), 1)
    BACKGROUND_CACHE = frame
    return BACKGROUND_CACHE.copy()


def draw_text(frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(frame, text, y, scale=0.85, color=(255, 255, 255), thickness=2):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(12, (SCREEN_WIDTH - text_w) // 2)
    draw_text(frame, text, (x, y + text_h // 2), scale, color, thickness)


def draw_prompt_bar(frame, text, y, color, scale=0.74, thickness=2):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x1 = max(24, (SCREEN_WIDTH - text_w) // 2 - 28)
    x2 = min(SCREEN_WIDTH - 24, (SCREEN_WIDTH + text_w) // 2 + 28)
    y1 = int(y - text_h // 2 - 20)
    y2 = int(y + text_h // 2 + 22)
    draw_round_rect(frame, (x1, y1), (x2, y2), (8, 12, 18), radius=8, alpha=0.88)
    draw_round_rect(frame, (x1, y1), (x2, y2), color, radius=8, thickness=1, alpha=0.68)
    draw_centered_text(frame, text, y - text_h // 2, scale, color, thickness)


def mix_color(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def _draw_round_rect_raw(frame, x1, y1, x2, y2, color, radius, thickness):
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius <= 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return

    if thickness < 0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        return

    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_round_rect(frame, pt1, pt2, color, radius=8, thickness=-1, alpha=1.0):
    x1, y1 = pt1
    x2, y2 = pt2
    x1 = int(np.clip(x1, 0, frame.shape[1] - 1))
    y1 = int(np.clip(y1, 0, frame.shape[0] - 1))
    x2 = int(np.clip(x2, x1 + 1, frame.shape[1]))
    y2 = int(np.clip(y2, y1 + 1, frame.shape[0]))

    if alpha >= 1.0:
        _draw_round_rect_raw(frame, x1, y1, x2, y2, color, radius, thickness)
        return

    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    _draw_round_rect_raw(overlay, 0, 0, x2 - x1 - 1, y2 - y1 - 1, color, radius, thickness)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_panel(frame, pt1, pt2, border_color=(86, 150, 170), fill_color=(12, 18, 28)):
    x1, y1 = pt1
    x2, y2 = pt2
    draw_round_rect(frame, (x1 + 8, y1 + 10), (x2 + 8, y2 + 10), (0, 0, 0), radius=8, alpha=0.24)
    draw_round_rect(frame, pt1, pt2, fill_color, radius=8, alpha=0.84)
    draw_round_rect(frame, pt1, pt2, border_color, radius=8, thickness=2, alpha=0.9)


def draw_pill(frame, text, center_x, baseline_y, color, scale=0.56):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x1 = int(center_x - text_w / 2 - 14)
    y1 = int(baseline_y - text_h - 10)
    x2 = int(center_x + text_w / 2 + 14)
    y2 = int(baseline_y + 10)
    draw_round_rect(frame, (x1, y1), (x2, y2), mix_color(color, (8, 12, 18), 0.58), radius=8, alpha=0.95)
    draw_round_rect(frame, (x1, y1), (x2, y2), color, radius=8, thickness=1, alpha=0.75)
    draw_text(frame, text, (x1 + 14, baseline_y), scale, (255, 255, 255), 1)


def draw_big_countdown(frame, progress, total_seconds, caption, color):
    if progress <= 0:
        return

    remaining = max(1, int(np.ceil(total_seconds * (1.0 - progress))))
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    radius = 104

    overlay = np.zeros_like(frame)
    cv2.circle(overlay, (center_x, center_y), radius + 44, color, -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 26)
    cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, dst=frame)

    cv2.circle(frame, (center_x, center_y), radius + 12, (8, 12, 18), -1, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y), radius + 12, color, 3, cv2.LINE_AA)
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), -90, 0, int(360 * progress), (255, 255, 255), 12, cv2.LINE_AA)

    number = str(remaining)
    (num_w, num_h), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 7)
    draw_text(frame, number, (center_x - num_w // 2, center_y + num_h // 2), 2.8, (255, 255, 255), 7)

    (cap_w, cap_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    draw_round_rect(
        frame,
        (center_x - cap_w // 2 - 18, center_y + radius + 34),
        (center_x + cap_w // 2 + 18, center_y + radius + cap_h + 58),
        (8, 12, 18),
        radius=8,
        alpha=0.9,
    )
    draw_text(frame, caption, (center_x - cap_w // 2, center_y + radius + cap_h + 48), 0.72, color, 2)


def draw_grid(frame, game, selected_cell, flash_cell, now):
    draw_panel(
        frame,
        (GRID_X - 24, GRID_Y - 24),
        (GRID_X + GRID_SIZE + 24, GRID_Y + GRID_SIZE + 24),
        border_color=(74, 128, 142),
        fill_color=(10, 14, 24),
    )

    for cell in range(9):
        row = cell // 3
        x1, y1, x2, y2 = cell_rect(cell)
        row_name, row_color = ROW_INFO[row]
        base = mix_color(row_color, (16, 21, 32), 0.78)
        fill = base
        border = mix_color(row_color, (255, 255, 255), 0.22)
        thickness = 2
        tile_alpha = 0.92

        if cell == selected_cell and game.state == "input":
            fill = mix_color(row_color, (255, 255, 255), 0.2)
            border = mix_color(row_color, (255, 255, 255), 0.68)
            thickness = 5
            tile_alpha = 1.0

        if cell == START_CELL and game.state in ("waiting", "game_over", "level_clear"):
            fill = mix_color(row_color, (255, 255, 255), 0.18)
            border = (120, 255, 160)
            thickness = 5
            tile_alpha = 1.0

        if game.state in ("waiting", "game_over") and cell in (NORMAL_DIFFICULTY_CELL, HARD_DIFFICULTY_CELL):
            difficulty_key = "normal" if cell == NORMAL_DIFFICULTY_CELL else "hard"
            difficulty = DIFFICULTY_SETTINGS[difficulty_key]
            fill = mix_color(difficulty["color"], (16, 21, 32), 0.58)
            border = difficulty["color"]
            thickness = 5 if game.difficulty == difficulty_key else 3
            tile_alpha = 1.0

        if cell == flash_cell:
            fill = mix_color(row_color, (255, 255, 255), 0.62)
            border = (255, 255, 255)
            thickness = 7
            tile_alpha = 1.0

        if game.feedback_cell == cell and now < game.feedback_until:
            fill = mix_color(game.feedback_color, (255, 255, 255), 0.08)
            border = (255, 255, 255)
            thickness = 7

        draw_round_rect(frame, (x1 + 5, y1 + 7), (x2 + 5, y2 + 7), (0, 0, 0), radius=8, alpha=0.22)
        draw_round_rect(frame, (x1, y1), (x2, y2), fill, radius=8, alpha=tile_alpha)
        draw_round_rect(frame, (x1 + 5, y1 + 5), (x2 - 5, y1 + 20), mix_color(row_color, (255, 255, 255), 0.35), radius=6, alpha=0.18)
        draw_round_rect(frame, (x1, y1), (x2, y2), border, radius=8, thickness=thickness, alpha=0.95)
        draw_pill(frame, row_name, x1 + 52, y1 + 34, row_color, scale=0.42)

        if game.hold_cell == cell and game.state == "input":
            progress = game.hold_progress
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = CELL_SIZE // 2 - 18
            cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * progress), (120, 255, 160), 10, cv2.LINE_AA)

        if cell == START_CELL and game.state in ("waiting", "game_over", "level_clear"):
            progress = game.next_level_hold_progress if game.state == "level_clear" else game.start_hold_progress
            label = "NEXT" if game.state == "level_clear" else "START"
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = CELL_SIZE // 2 - 20
            cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * progress), (120, 255, 160), 10, cv2.LINE_AA)
            draw_text(frame, label, (x1 + 64, y1 + CELL_SIZE // 2 + 8), 0.74, (255, 255, 255), 2)

        if game.state in ("waiting", "game_over") and cell in (NORMAL_DIFFICULTY_CELL, HARD_DIFFICULTY_CELL):
            difficulty_key = "normal" if cell == NORMAL_DIFFICULTY_CELL else "hard"
            difficulty = DIFFICULTY_SETTINGS[difficulty_key]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            label = difficulty["label"].upper()
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            draw_text(frame, label, (center[0] - label_w // 2, center[1] + label_h // 2), 0.72, (255, 255, 255), 2)

            if game.difficulty == difficulty_key:
                draw_pill(frame, "SELECTED", center[0], y2 - 24, difficulty["color"], scale=0.42)

            if game.difficulty_hold_cell == cell:
                radius = CELL_SIZE // 2 - 18
                cv2.ellipse(
                    frame,
                    center,
                    (radius, radius),
                    -90,
                    0,
                    int(360 * game.difficulty_hold_progress),
                    (255, 255, 255),
                    9,
                    cv2.LINE_AA,
                )


def draw_player_marker(frame, person, selected_cell, game, now):
    if not person.present or selected_cell is None:
        return

    row = selected_cell // 3
    color = ROW_INFO[row][1]
    position = person_grid_position(person)
    if position is None:
        return
    cx, cy = position
    pulse = 1.0 + 0.08 * np.sin(now * 5.5)
    radius = int(PLAYER_MARKER_RADIUS * pulse)
    cell_cx, cell_cy = cell_center(selected_cell)

    cv2.line(frame, (cx, cy), (cell_cx, cell_cy), mix_color(color, (255, 255, 255), 0.46), 2, cv2.LINE_AA)
    cv2.circle(frame, (cell_cx, cell_cy), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cell_cx, cell_cy), 15, color, 2, cv2.LINE_AA)

    glow_margin = radius * 4
    x0 = max(0, cx - glow_margin)
    x1 = min(SCREEN_WIDTH, cx + glow_margin)
    y0 = max(0, cy - glow_margin)
    y1 = min(SCREEN_HEIGHT, cy + glow_margin)
    if x0 < x1 and y0 < y1:
        glow = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cv2.circle(glow, (cx - x0, cy - y0), radius * 3, color, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 18)
        roi = frame[y0:y1, x0:x1]
        cv2.addWeighted(glow, 0.75, roi, 1.0, 0, dst=roi)

    cv2.circle(frame, (cx, cy), radius + 11, (8, 12, 18), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 8, mix_color(color, (255, 255, 255), 0.34), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(10, radius // 2), mix_color(color, (255, 255, 255), 0.26), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx - radius // 3, cy - radius // 3), max(5, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)

    if game.state == "input" and game.center_locked_until_exit and selected_cell == START_CELL:
        draw_centered_text(frame, "leave center", cy + 44, 0.42, (255, 255, 255), 1)
    else:
        label = f"{person.distance_m:.2f} m"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        draw_round_rect(frame, (cx - text_w // 2 - 10, cy + radius + 10), (cx + text_w // 2 + 10, cy + radius + text_h + 28), (8, 12, 18), radius=8, alpha=0.92)
        draw_text(frame, label, (cx - text_w // 2, cy + radius + text_h + 18), 0.54, (255, 255, 255), 1)


def draw_header(frame, game, fps_value):
    difficulty = game.difficulty_settings()
    draw_round_rect(frame, (18, 14), (SCREEN_WIDTH - 18, 66), (7, 11, 18), radius=8, alpha=0.92)
    draw_round_rect(frame, (18, 14), (SCREEN_WIDTH - 18, 66), (70, 132, 150), radius=8, thickness=1, alpha=0.75)
    draw_text(frame, "Sequence Memory", (34, 48), 0.9, (92, 230, 255), 2)
    draw_pill(frame, f"Level {game.level}", 428, 48, (95, 235, 145), scale=0.56)
    draw_pill(frame, f"Length {len(game.sequence)}", 566, 48, (255, 180, 90), scale=0.56)
    draw_pill(frame, f"Best {game.best_level}", 724, 48, (255, 115, 215), scale=0.56)
    draw_pill(frame, difficulty["label"], 858, 48, difficulty["color"], scale=0.56)
    if SHOW_FPS_COUNTER:
        cap_text = "uncapped" if MAX_PROGRAM_FPS <= 0 else f"cap {MAX_PROGRAM_FPS:g}"
        draw_text(frame, f"FPS {fps_value:4.1f}  {cap_text}", (SCREEN_WIDTH - 304, 45), 0.56, (135, 255, 170), 1)


def draw_status(frame, game, person, selected_cell):
    panel_x = 34
    panel_y = 116
    panel_w = 250
    panel_h = 258
    draw_panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_color=(78, 142, 158), fill_color=(9, 13, 23))
    draw_text(frame, "Player", (panel_x + 22, panel_y + 35), 0.62, (92, 230, 255), 2)

    if person.present:
        draw_text(frame, f"{person.distance_m:.2f} m", (panel_x + 22, panel_y + 82), 1.02, (135, 255, 170), 2)
        draw_text(frame, f"YOLO {person.confidence:.2f}", (panel_x + 24, panel_y + 120), 0.56, (225, 235, 240), 1)
        if selected_cell is not None:
            row = selected_cell // 3
            col = selected_cell % 3
            draw_round_rect(frame, (panel_x + 20, panel_y + 138), (panel_x + panel_w - 20, panel_y + 204), mix_color(ROW_INFO[row][1], (8, 12, 18), 0.62), radius=8, alpha=0.9)
            draw_text(frame, f"Row {row + 1}", (panel_x + 34, panel_y + 164), 0.62, ROW_INFO[row][1], 2)
            draw_text(frame, f"Column {col + 1}", (panel_x + 34, panel_y + 194), 0.62, (255, 255, 255), 2)
            if game.state == "input":
                draw_text(frame, "hold to select", (panel_x + 24, panel_y + 238), 0.56, (180, 220, 235), 1)
    else:
        draw_text(frame, "--.-- m", (panel_x + 22, panel_y + 82), 1.02, (180, 180, 180), 2)
        draw_text(frame, "waiting for human", (panel_x + 24, panel_y + 124), 0.56, (225, 235, 240), 1)


def draw_prompt(frame, game, person):
    difficulty = game.difficulty_settings()
    if game.state == "waiting":
        draw_prompt_bar(frame, f"{difficulty['label']} selected - hold left/right for 3 seconds to change, center to start", 640, difficulty["color"], 0.62, 2)
    elif game.state == "showing":
        draw_prompt_bar(frame, "Watch the sequence", 640, (255, 235, 150), 0.8, 2)
    elif game.state == "input":
        if game.center_locked_until_exit:
            draw_prompt_bar(frame, "Move out of the center square to begin", 640, (255, 255, 255), 0.74, 2)
        else:
            draw_prompt_bar(frame, f"Repeat step {game.input_index + 1} of {len(game.sequence)}", 640, (255, 255, 255), 0.8, 2)
    elif game.state == "level_clear":
        draw_prompt_bar(frame, "Correct - stand in the center square for 3 seconds", 640, (120, 255, 160), 0.7, 2)
    elif game.state == "game_over":
        draw_prompt_bar(frame, f"Wrong square - {difficulty['label']} selected, center for 5 seconds to restart", 640, (100, 180, 255), 0.62, 2)

    if not person.present and game.state in ("waiting", "game_over", "input", "showing", "level_clear"):
        draw_centered_text(frame, "Stand between 0.5 m and 3.5 m from the camera", 600, 0.62, (230, 230, 230), 1)


def draw_countdown_overlay(frame, game):
    if not SHOW_BIG_COUNTDOWN:
        return

    if game.state in ("waiting", "game_over") and game.difficulty_hold_progress > 0:
        difficulty_key = "normal" if game.difficulty_hold_cell == NORMAL_DIFFICULTY_CELL else "hard"
        difficulty = DIFFICULTY_SETTINGS[difficulty_key]
        draw_big_countdown(
            frame,
            game.difficulty_hold_progress,
            DIFFICULTY_SELECT_HOLD_SECONDS,
            f"select {difficulty['label']}",
            difficulty["color"],
        )
    elif game.state in ("waiting", "game_over"):
        draw_big_countdown(frame, game.start_hold_progress, START_HOLD_SECONDS, "hold center to start", (120, 255, 160))
    elif game.state == "level_clear":
        draw_big_countdown(frame, game.next_level_hold_progress, NEXT_LEVEL_HOLD_SECONDS, "hold center for next level", (120, 255, 160))


def draw_row_guide(frame):
    guide_x = GRID_X + GRID_SIZE + 52
    guide_y = GRID_Y + 18
    draw_panel(frame, (guide_x - 20, guide_y - 28), (guide_x + 246, guide_y + 236), border_color=(78, 142, 158), fill_color=(9, 13, 23))
    draw_text(frame, "Depth rows", (guide_x, guide_y), 0.62, (90, 225, 255), 2)
    draw_text(frame, f"size {SQUARE_SIZE_DISTANSE}: {SQUARE_PROFILE['name']}", (guide_x, guide_y + 30), 0.44, (225, 235, 240), 1)
    labels = [
        ("near", f"{ACTIVE_NEAR_M:.1f}-{NEAR_ROW_MAX_M:.1f} m", ROW_INFO[0][1]),
        ("middle", f"{NEAR_ROW_MAX_M:.1f}-{MIDDLE_ROW_MAX_M:.1f} m", ROW_INFO[1][1]),
        ("far", f"{MIDDLE_ROW_MAX_M:.1f}-{ACTIVE_FAR_M:.1f} m", ROW_INFO[2][1]),
    ]
    for i, (name, depth_text, color) in enumerate(labels):
        y = guide_y + 70 + i * 58
        draw_round_rect(frame, (guide_x, y - 28), (guide_x + 204, y + 16), mix_color(color, (8, 12, 18), 0.36), radius=8, alpha=0.96)
        draw_round_rect(frame, (guide_x, y - 28), (guide_x + 204, y + 16), color, radius=8, thickness=1, alpha=0.85)
        draw_text(frame, name, (guide_x + 12, y), 0.54, (255, 255, 255), 1)
        draw_text(frame, depth_text, (guide_x + 96, y), 0.46, (255, 255, 255), 1)


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
    rgb = prepare_rgb_for_detection(rgb_frame)
    if rgb is None:
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        draw_text(frame, "Waiting for RGB frame", (42, CAMERA_HEIGHT // 2), 0.72)
        return frame
    return rgb


def draw_debug_windows(rgb_frame, depth_m, person):
    rgb_debug = rgb_to_debug_display(rgb_frame)
    depth_debug = depth_to_debug_display(depth_m)

    if person.present:
        marker = (
            int(person.x_norm * CAMERA_WIDTH),
            int(person.y_norm * CAMERA_HEIGHT),
        )
        if person.bbox is not None:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(rgb_debug, (x1, y1), (x2, y2), (90, 255, 130), 2, cv2.LINE_AA)
            cv2.rectangle(depth_debug, (x1, y1), (x2, y2), (90, 255, 130), 2, cv2.LINE_AA)
        cv2.circle(rgb_debug, marker, 14, (90, 255, 130), 2, cv2.LINE_AA)
        cv2.circle(depth_debug, marker, 14, (90, 255, 130), 2, cv2.LINE_AA)
        draw_text(rgb_debug, f"{person.distance_m:.2f} m", (marker[0] + 18, marker[1] - 12), 0.58, (90, 255, 130), 1)
        draw_text(depth_debug, f"{person.distance_m:.2f} m", (marker[0] + 18, marker[1] - 12), 0.58, (90, 255, 130), 1)

    cv2.imshow(RGB_DEBUG_WINDOW, rgb_debug)
    cv2.imshow(DEPTH_DEBUG_WINDOW, depth_debug)


def close_debug_windows():
    for window_name in (RGB_DEBUG_WINDOW, DEPTH_DEBUG_WINDOW):
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def main():
    camera = HP60SDKCamera(sdk_root=SDK_ROOT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    camera.start()
    yolo_detector = YOLOPersonDetector()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)

    game = SequenceGame()
    smoother = SmoothPerson()
    fps_meter = FPSMeter()
    debug_enabled = RGB_DEPTH_DEBUG
    fullscreen = False

    try:
        while True:
            loop_start_time = time.perf_counter()
            now = time.time()
            fps_value = fps_meter.update()

            rgb_frame, raw_depth = camera.get_latest_frames()
            depth_m = depth_to_meters(raw_depth)
            if yolo_detector.ready:
                measured = yolo_detector.detect(rgb_frame, depth_m)
            elif USE_YOLO_PERSON_DETECTION:
                measured = PersonState()
            else:
                measured = PersonState()
            person = smoother.update(measured)
            selected_cell = selected_cell_from_person(person)
            game.update_difficulty_control(now, selected_cell)
            game.update_start_control(now, selected_cell)

            flash_cell = game.current_flash_cell(now)
            game.update(now, selected_cell)

            frame = make_background(now)
            draw_grid(frame, game, selected_cell, flash_cell, now)
            draw_player_marker(frame, person, selected_cell, game, now)
            draw_row_guide(frame)
            draw_status(frame, game, person, selected_cell)
            draw_header(frame, game, fps_value)
            draw_prompt(frame, game, person)
            draw_countdown_overlay(frame, game)
            if USE_YOLO_PERSON_DETECTION and not yolo_detector.ready:
                draw_centered_text(frame, yolo_detector.error, 590, 0.62, (100, 200, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)
            if debug_enabled:
                draw_debug_windows(rgb_frame, depth_m, person)
            else:
                close_debug_windows()

            limit_program_fps(loop_start_time)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                if game.state in ("waiting", "game_over"):
                    game.start(time.time())
            if key == ord("d"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    close_debug_windows()
            if key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            if key == ord("r"):
                game = SequenceGame()
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
