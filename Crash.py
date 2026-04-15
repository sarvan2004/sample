import cv2
import numpy as np
import csv
from datetime import datetime


class CrashDetector:
    def __init__(self, model, roi_points=None, resize=(480, 360)):

        self.model = model

        self.roi_points = roi_points
        self.roi_defined = roi_points is not None

        self.RESIZE_WIDTH, self.RESIZE_HEIGHT = resize

        # PARAMETERS
        self.SPIKE_THRESHOLD = 1.0
        self.VAR_THRESHOLD = 1.5
        self.CONFLICT_THRESHOLD = 2
        self.DROP_THRESHOLD = 1.0
        self.CRASH_FRAME_THRESHOLD = 3

        # STATE
        self.prev_gray = None
        self.prev_mean_mag = 0
        self.crash_counter = 0
        self.frame_id = 0

        # ================= LOG =================
        self.csv_file = open("crash_log.csv", mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp", "Frame",
            "MeanMag", "Variance",
            "Spike", "Drop",
            "Conflict", "Crash"
        ])

        # ================= RECORD =================
        self.recording = False
        self.out = None
        self.no_crash_frames = 0
        self.stop_threshold = 30   # stop recording after no crash

    # =========================================================
    def direction_conflict_local(self, ang, mag):
        mask = mag > 1.0
        angles = ang[mask] * 180 / np.pi

        if len(angles) < 50:
            return 0

        hist, _ = np.histogram(angles, bins=8, range=(0,360))
        return np.sum(hist > (0.15 * np.max(hist)))

    # =========================================================
    def process(self, frame):

        self.frame_id += 1

        frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = self.model(frame, verbose=False)

        vehicle_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]

                if label in ["car","motorcycle","bus","truck"]:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx,cy = (x1+x2)//2,(y1+y2)//2

                    if self.roi_defined:
                        inside = cv2.pointPolygonTest(
                            np.array(self.roi_points), (cx,cy), False
                        ) >= 0
                    else:
                        inside = True

                    if inside:
                        vehicle_detected = True

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, frame

        # ================= OPTICAL FLOW =================
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5,2,9,2,3,1.1,0
        )

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        mean_mag = np.mean(mag)
        var_mag = np.var(mag)
        spike = abs(mean_mag - self.prev_mean_mag)
        drop = self.prev_mean_mag - mean_mag

        conflict = self.direction_conflict_local(ang, mag)

        crash = False

        if vehicle_detected:
            if conflict >= self.CONFLICT_THRESHOLD and spike > self.SPIKE_THRESHOLD:
                crash = True
            elif conflict >= self.CONFLICT_THRESHOLD and var_mag > self.VAR_THRESHOLD:
                crash = True
            elif spike > self.SPIKE_THRESHOLD and var_mag > self.VAR_THRESHOLD:
                crash = True
            elif drop > self.DROP_THRESHOLD:
                crash = True

        # ================= TEMPORAL FILTER =================
        if crash:
            self.crash_counter += 1
        else:
            self.crash_counter = max(0, self.crash_counter - 1)

        final_crash = self.crash_counter >= self.CRASH_FRAME_THRESHOLD

        # ================= LOG =================
        if final_crash and self.crash_counter == self.CRASH_FRAME_THRESHOLD:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.csv_writer.writerow([
                timestamp,self.frame_id,
                round(mean_mag,3),round(var_mag,3),
                round(spike,3),round(drop,3),
                conflict,int(final_crash)
            ])

        # ================= RECORDING =================
        if final_crash:
            self.no_crash_frames = 0

            if not self.recording:
                filename = f"crash_{datetime.now().strftime('%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(
                    filename, fourcc, 20,
                    (self.RESIZE_WIDTH, self.RESIZE_HEIGHT)
                )
                self.recording = True

            if self.out is not None:
                self.out.write(frame)

        else:
            if self.recording:
                self.no_crash_frames += 1

                if self.no_crash_frames > self.stop_threshold:
                    self.recording = False
                    self.out.release()
                    self.out = None

        # ================= UPDATE =================
        self.prev_gray = gray
        self.prev_mean_mag = mean_mag

        return final_crash, frame

    # =========================================================
    def release(self):
        self.csv_file.close()
        if self.out is not None:
            self.out.release()