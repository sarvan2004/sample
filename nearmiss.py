import cv2
import numpy as np
from collections import defaultdict, deque
from scipy.spatial import distance as dist
import os


class NearMissDetector:
    def __init__(self, model, roi_points=None, resize=(640,360)):

        self.model = model

        self.roi_points = roi_points
        self.zone_defined = roi_points is not None
        self.ZONE = np.array(roi_points, np.int32) if roi_points else None

        self.RESIZE_WIDTH, self.RESIZE_HEIGHT = resize

        self.objects = {}
        self.next_id = 0
        self.trajectories = defaultdict(lambda: deque(maxlen=10))
        self.speeds = {}

        self.prev_gray = None
        self.prev_energy = 0

        self.prev_distances = {}
        self.frame_buffer = deque(maxlen=90)

        self.event_count = 0
        self.saving = False
        self.save_frames_left = 0
        self.out = None

        os.makedirs("near_miss_clips", exist_ok=True)

    # ================= DETECTION =================
    def detect(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)[0]
        boxes = []

        if results.boxes is None:
            return boxes

        for r in results.boxes:
            cls = int(r.cls[0])
            if cls in [2,3,5,7]:
                x1,y1,x2,y2 = map(int, r.xyxy[0])
                boxes.append((x1,y1,x2-x1,y2-y1))

        return boxes

    # ================= TRACKER =================
    def update_tracker(self, boxes):
        new_objects = {}

        # 🔥 FIX 1: handle empty detections
        if len(boxes) == 0:
            self.objects = {}
            return

        centroids = np.array(
            [(b[0]+b[2]//2, b[1]+b[3]//2) for b in boxes],
            dtype=np.float32
        )

        # 🔥 FIX 2: ensure valid shape
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            return

        if len(self.objects) == 0:
            for c in centroids:
                new_objects[self.next_id] = c
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()), dtype=np.float32)

            # 🔥 FIX 3: safeguard before cdist
            if len(object_centroids) == 0:
                self.objects = {}
                return

            D = dist.cdist(object_centroids, centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for r,c in zip(rows,cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r][c] < 60:
                    new_objects[object_ids[r]] = centroids[c]
                    used_rows.add(r)
                    used_cols.add(c)

            for i in range(len(centroids)):
                if i not in used_cols:
                    new_objects[self.next_id] = centroids[i]
                    self.next_id += 1

        self.objects = new_objects

    # ================= SPEED =================
    def compute_speed(self):
        for obj_id, c in self.objects.items():
            self.trajectories[obj_id].append(c)

            if len(self.trajectories[obj_id]) >= 2:
                p1 = self.trajectories[obj_id][-2]
                p2 = self.trajectories[obj_id][-1]
                self.speeds[obj_id] = np.linalg.norm(np.array(p2)-np.array(p1))

    # ================= OPTICAL FLOW =================
    def cetm_signal(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5,2,9,2,3,1.1,0
        )

        mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])

        energy = np.mean(mag**2)
        drop = self.prev_energy - energy

        self.prev_gray = gray
        self.prev_energy = energy

        return energy > 1.2 and drop > 0.3

    # ================= MAIN =================
    def process(self, frame):

        frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
        self.frame_buffer.append(frame.copy())

        boxes = self.detect(frame)
        self.update_tracker(boxes)
        self.compute_speed()

        cetm_flag = self.cetm_signal(frame)

        ids = list(self.objects.keys())

        # 🔥 FIX 4: avoid pair crash
        if len(ids) < 2:
            return False, frame

        near_miss = False

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1,id2 = ids[i], ids[j]
                pair = (id1,id2)

                c1 = self.objects[id1]
                c2 = self.objects[id2]

                if self.zone_defined:
                    if cv2.pointPolygonTest(self.ZONE, tuple(map(int,c1)), False) < 0:
                        continue
                    if cv2.pointPolygonTest(self.ZONE, tuple(map(int,c2)), False) < 0:
                        continue

                d = np.linalg.norm(np.array(c1)-np.array(c2))
                v1 = self.speeds.get(id1,0)
                v2 = self.speeds.get(id2,0)

                if v1 < 2 and v2 < 2:
                    continue

                if pair in self.prev_distances:
                    if self.prev_distances[pair] < 40 and d > self.prev_distances[pair] + 25:

                        near_miss = True

                        if not self.saving:
                            self.event_count += 1

                            filename = f"near_miss_clips/near_miss_{self.event_count}.avi"

                            self.out = cv2.VideoWriter(
                                filename,
                                cv2.VideoWriter_fourcc(*'XVID'),
                                20,
                                (self.RESIZE_WIDTH,self.RESIZE_HEIGHT)
                            )

                            for f in self.frame_buffer:
                                self.out.write(f)

                            self.saving = True
                            self.save_frames_left = 60

                self.prev_distances[pair] = d

        if self.saving and self.out:
            self.out.write(frame)
            self.save_frames_left -= 1

            if self.save_frames_left <= 0:
                self.saving = False
                self.out.release()
                self.out = None

        return near_miss, frame

    def release(self):
        if self.out:
            self.out.release()