"""
object_tracking.py  -  Real-time Object Detection and Tracking

  Video input : OpenCV (webcam or video file)
  Detection   : YOLOv8 pre-trained on COCO (80 classes)
  Tracking    : SORT (Kalman filter + Hungarian algorithm), implemented below
  Output      : bounding boxes, class labels, confidence, tracking IDs, trails, FPS

Examples
  python object_tracking.py                                  # webcam 0
  python object_tracking.py --source video.mp4
  python object_tracking.py --source video.mp4 --save output.mp4
  python object_tracking.py --classes person car --conf 0.4
  python object_tracking.py --model yolov8s.pt --device cpu

Keys while running:  q / ESC = quit,  p = pause,  s = save screenshot
"""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# =====================================================================
# 1. SORT TRACKER (Simple Online and Realtime Tracking)
# =====================================================================
def iou_matrix(boxes_a, boxes_b):
    """IoU between every box in A and every box in B. Boxes are [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]    # (N, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]    # (1, M, 4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def box_to_z(box):
    """[x1, y1, x2, y2] -> [center_x, center_y, area, aspect_ratio]"""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / float(h + 1e-9)])


def x_to_box(x):
    """Kalman state -> [x1, y1, x2, y2]"""
    w = np.sqrt(max(x[2] * x[3], 1e-9))
    h = x[2] / (w + 1e-9)
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0])


class KalmanBoxTracker:
    """One tracked object. State = [cx, cy, area, ratio, v_cx, v_cy, v_area]."""
    _next_id = 1

    def __init__(self, box, cls_id, conf):
        # constant-velocity model
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0       # high uncertainty about the initial velocity
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.cls_id = cls_id
        self.conf = conf
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, cls_id, conf):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls_id = cls_id
        self.conf = conf

        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def box(self):
        return x_to_box(self.x)


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age              # frames to keep a lost track alive
        self.min_hits = min_hits            # detections needed before a track is shown
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._next_id = 1

    def update(self, detections):
        """
        detections: numpy array (N, 6) -> [x1, y1, x2, y2, conf, cls_id]
        returns   : numpy array (M, 7) -> [x1, y1, x2, y2, track_id, cls_id, conf]
        """
        self.frame_count += 1

        # 1) predict where the existing tracks should be now
        predicted = []
        for t in self.trackers:
            box = t.predict()
            predicted.append(box)
        valid = [i for i, b in enumerate(predicted) if not np.any(np.isnan(b))]
        self.trackers = [self.trackers[i] for i in valid]
        predicted = [predicted[i] for i in valid]

        # 2) match detections to tracks (Hungarian algorithm on IoU)
        matches, unmatched_dets = [], list(range(len(detections)))
        if len(self.trackers) and len(detections):
            iou = iou_matrix(detections[:, :4], predicted)          # (dets, tracks)
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = []
            matched_rows = set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    matched_rows.add(r)
            unmatched_dets = [d for d in range(len(detections)) if d not in matched_rows]

        # 3) update matched tracks
        for det_idx, trk_idx in matches:
            d = detections[det_idx]
            self.trackers[trk_idx].update(d[:4], int(d[5]), float(d[4]))

        # 4) start new tracks for unmatched detections
        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.trackers.append(KalmanBoxTracker(d[:4], int(d[5]), float(d[4])))

        # 5) collect output and remove dead tracks
        results = []
        for t in self.trackers:
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if t.time_since_update < 1 and confirmed:
                results.append([*t.box(), t.id, t.cls_id, t.conf])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return np.array(results, dtype=np.float32) if results else np.empty((0, 7), dtype=np.float32)


# =====================================================================
# 2. DETECTOR (YOLO)
# =====================================================================
class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.35, iou=0.5,
                 device=None, class_names=None, img_size=640):
        from ultralytics import YOLO

        self.model = YOLO(model_path)               # downloads the weights on first run
        self.names = self.model.names               # {0: 'person', 1: 'bicycle', ...}
        self.conf = conf
        self.iou = iou
        self.device = device
        self.img_size = img_size
        self.class_filter = None

        if class_names:
            name_to_id = {n.lower(): i for i, n in self.names.items()}
            ids = []
            for name in class_names:
                if name.lower() not in name_to_id:
                    raise ValueError(f"Unknown class '{name}'. Available: {sorted(name_to_id)}")
                ids.append(name_to_id[name.lower()])
            self.class_filter = ids

    def detect(self, frame):
        """Returns array (N, 6): [x1, y1, x2, y2, conf, cls_id]"""
        result = self.model(frame, conf=self.conf, iou=self.iou, classes=self.class_filter,
                            imgsz=self.img_size, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()[:, None]
        cls = result.boxes.cls.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf, cls]).astype(np.float32)


# =====================================================================
# 3. DRAWING
# =====================================================================
def color_for_id(track_id):
    """A stable, bright colour for each track ID."""
    rng = np.random.RandomState(int(track_id) * 7919 % (2 ** 31))
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_tracks(frame, tracks, names, trails, show_trails=True):
    for x1, y1, x2, y2, tid, cls_id, conf in tracks:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid, cls_id = int(tid), int(cls_id)
        color = color_for_id(tid)
        label = f"ID {tid} {names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # trail of the object's centre points
        centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        trails[tid].append(centre)
        if show_trails:
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                thickness = max(1, int(np.sqrt(i) * 0.8))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)


def draw_panel(frame, fps, frame_no, active, unique_counts):
    """Info panel in the top-left corner."""
    lines = [f"FPS: {fps:.1f}", f"Frame: {frame_no}", f"Tracking now: {active}"]
    for name, ids in sorted(unique_counts.items(), key=lambda kv: -len(kv[1]))[:6]:
        lines.append(f"Total {name}: {len(ids)}")

    h = 12 + 22 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (240, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


# =====================================================================
# 4. MAIN LOOP
# =====================================================================
def open_source(source):
    """Webcam index (like 0) or a video file path."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def run(args):
    cap = open_source(args.source)
    is_webcam = str(args.source).isdigit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1 or src_fps > 120:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    print(f"Source: {args.source}  ({width}x{height} @ {src_fps:.0f} fps)")

    print("Loading model...")
    detector = YoloDetector(args.model, args.conf, args.iou, args.device,
                            args.classes, args.img_size)
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.track_iou)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (width, height))
        print(f"Saving output to: {args.save}")

    trails = defaultdict(lambda: deque(maxlen=args.trail_length))
    unique_counts = defaultdict(set)            # class name -> set of track IDs ever seen
    shots_dir = Path("screenshots")

    frame_no, fps, paused = 0, 0.0, False
    last_time = time.time()
    print("Running. Press q or ESC to quit, p to pause, s to save a screenshot.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video / camera stream.")
                break
            frame_no += 1
            if is_webcam and args.mirror:
                frame = cv2.flip(frame, 1)

            # detect -> track -> draw
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            for t in tracks:
                unique_counts[detector.names[int(t[5])]].add(int(t[4]))
            draw_tracks(frame, tracks, detector.names, trails, not args.no_trails)

            # smoothed FPS
            now = time.time()
            instant = 1.0 / max(now - last_time, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
            last_time = now

            draw_panel(frame, fps, frame_no, len(tracks), unique_counts)
            if total_frames:
                cv2.putText(frame, f"{frame_no}/{total_frames}", (width - 140, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if writer:
                writer.write(frame)

        if not args.no_show:
            cv2.imshow("Object Detection and Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                shots_dir.mkdir(exist_ok=True)
                path = shots_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Screenshot saved: {path}")
        elif total_frames and frame_no % 50 == 0:
            print(f"Processed {frame_no}/{total_frames} frames ({fps:.1f} fps)")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n----- Summary -----")
    print(f"Frames processed: {frame_no}")
    for name, ids in sorted(unique_counts.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name}: {len(ids)} unique object(s) tracked")


def parse_args():
    p = argparse.ArgumentParser(description="Object detection and tracking (YOLOv8 + SORT)")
    p.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights: yolov8n.pt (fastest) ... yolov8s/m/l/x.pt (more accurate)")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--img-size", type=int, default=640, help="YOLO input size (smaller = faster)")
    p.add_argument("--device", default=None, help="'cpu', '0' (first GPU), 'mps' (Apple)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="only detect these classes, e.g. --classes person car")
    p.add_argument("--max-age", type=int, default=30, help="frames to keep a lost track")
    p.add_argument("--min-hits", type=int, default=3, help="hits before a track is shown")
    p.add_argument("--track-iou", type=float, default=0.3, help="IoU needed to match a track")
    p.add_argument("--trail-length", type=int, default=30, help="length of motion trails")
    p.add_argument("--no-trails", action="store_true", help="do not draw motion trails")
    p.add_argument("--mirror", action="store_true", help="flip webcam image horizontally")
    p.add_argument("--save", default=None, help="save the output video, e.g. output.mp4")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())