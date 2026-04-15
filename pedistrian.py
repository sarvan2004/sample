import cv2
import csv
import os
import numpy as np


class PedestrianViolationDetector:
    def __init__(self, model, roi_points=None, fps=30):

        self.model = model

        self.roi_points = roi_points
        self.roi_defined = roi_points is not None

        if self.roi_defined:
            # ✅ UPDATED: support multiple polygons
            self.polygons = [np.array(p, dtype=np.int32) for p in roi_points]

        # ================= LOG =================
        self.log_file = open("pedestrian_log.csv", "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["time", "vehicle_id"])

        # ================= MEMORY =================
        self.last_event = -9999
        self.cooldown = int(5 * fps)

        # ================= CLIPS =================
        os.makedirs("ped_clips", exist_ok=True)

        self.frame_buffer = []
        self.buffer_size = int(3 * fps)

        self.active_clips = []
        self.clip_id = 0

        self.fps = fps
        self.frame_idx = 0

    # =========================================================
    def inside_roi(self, bbox):
        if not self.roi_defined:
            return True

        x1,y1,x2,y2 = bbox
        cx,cy = int((x1+x2)/2), int((y1+y2)/2)

        # ✅ UPDATED: check across all polygons
        for poly in self.polygons:
            if cv2.pointPolygonTest(poly, (cx,cy), False) >= 0:
                return True

        return False

    # =========================================================
    def process(self, frame):

        self.frame_idx += 1

        results = self.model.track(frame, persist=True, verbose=False)

        pedestrians = []
        vehicles = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:

                if box.id is None:
                    continue

                cls = int(box.cls[0])
                tid = int(box.id[0])

                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                bbox = (x1,y1,x2,y2)

                if self.inside_roi(bbox):
                    if cls == 0:
                        pedestrians.append(tid)
                    elif cls in [2,3,5,7]:
                        vehicles.append(tid)

        # ================= BUFFER =================
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # ================= VIOLATION =================
        violators = []

        if pedestrians and vehicles:
            violators = vehicles

        # ================= EVENT =================
        if violators:

            if self.frame_idx - self.last_event > self.cooldown:

                timestamp = self.frame_idx
                rep_vid = violators[0]

                self.writer.writerow([timestamp, rep_vid])
                print(f"[PED] Frame {timestamp}, Vehicle {rep_vid}")

                self.last_event = self.frame_idx

                # clip start
                self.active_clips.append({
                    "frames": self.frame_buffer.copy(),
                    "end": self.frame_idx + int(3 * self.fps),
                    "id": self.clip_id
                })

                self.clip_id += 1

        # ================= SAVE CLIPS =================
        for clip in self.active_clips[:]:

            clip["frames"].append(frame.copy())

            if self.frame_idx >= clip["end"]:

                path = f"ped_clips/clip_{clip['id']}.mp4"

                out = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (frame.shape[1], frame.shape[0])
                )

                for f in clip["frames"]:
                    out.write(f)

                out.release()
                self.active_clips.remove(clip)

        return len(violators) > 0, frame

    # =========================================================
    def release(self):
        self.log_file.close()