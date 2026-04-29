import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime

DISTANCE = 20
SPEED_LIMIT = 30

class SpeedDetector:
    def __init__(self, model, line_pairs, frame_size=(960,540)):
        self.model = model
        self.line_pairs = line_pairs
        self.frame_size = frame_size

        self.transforms = {}
        for k, (l1, l2) in line_pairs.items():
            self.transforms[k] = self.get_transform(l1, l2)

        self.state = {
            "prev": {},
            "entry_fwd": {k:{} for k in line_pairs},
            "entry_rev": {k:{} for k in line_pairs},
            "triggered": set()
        }

        os.makedirs("speed_output", exist_ok=True)

    # =============================
    def get_transform(self, l1, l2):
        src = np.array([l1[0], l1[1], l2[1], l2[0]], dtype=np.float32)
        dst = np.array([[0,0],[20,0],[20,1000],[0,1000]], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def transform_point(self, M, pt):
        pts = np.array([pt], dtype=np.float32).reshape(-1,1,2)
        return cv2.perspectiveTransform(pts, M).reshape(-1,2)[0]

    def write_csv(self, road, row):
        path = f"speed_output/{road}.csv"
        file_exists = os.path.exists(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp","road","vehicle","speed"])
            writer.writerow(row)

    # =============================
    def process(self, frame):

        frame = cv2.resize(frame, self.frame_size)

        results = self.model(frame, verbose=False)[0]

        speed_flag = False
        now = time.time()

        if results.boxes is None:
            return speed_flag, frame

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in [2,3,5,7]:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx,cy = int((x1+x2)/2), int(y2)

            for pname,(l1,l2) in self.line_pairs.items():

                M = self.transforms[pname]
                ty = self.transform_point(M, (cx,cy))[1]

                line1_ty = np.mean([self.transform_point(M, p)[1] for p in l1])
                line2_ty = np.mean([self.transform_point(M, p)[1] for p in l2])

                tid = id(box)  # simple tracking substitute

                if tid in self.state["prev"]:
                    prev = self.state["prev"][tid]
                    crossed1 = (prev-line1_ty)*(ty-line1_ty)<=0
                    crossed2 = (prev-line2_ty)*(ty-line2_ty)<=0
                else:
                    crossed1=crossed2=False

                # FORWARD
                if crossed1:
                    self.state["entry_fwd"][pname][tid] = now

                if crossed2 and tid in self.state["entry_fwd"][pname]:
                    t = now - self.state["entry_fwd"][pname][tid]

                    if 0.3 < t < 5:
                        speed = (DISTANCE/t)*3.6
                        speed_flag = True

                        self.write_csv(pname, [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pname,
                            "vehicle",
                            int(speed)
                        ])

                        cv2.putText(frame, f"{int(speed)} km/h",
                                    (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                self.state["prev"][tid] = ty

        return speed_flag, frame

    def release(self):
        pass