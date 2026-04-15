import cv2
import csv
from datetime import datetime
from ultralytics import YOLO


class EmergencyDetector:
    def __init__(self, model_path="best.pt", frame_skip=6):

        # ================= MODEL =================
        self.model = YOLO(model_path)

        # ================= FRAME CONTROL =================
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.serial_no = 1

        # ================= RECORDING =================
        self.recording = False
        self.out = None
        self.no_detection_frames = 0
        self.stop_threshold = 30  # same as original

        # ================= CSV =================
        self.csv_file = open("detections.csv", mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["S.No", "Vehicle Type", "Timestamp"])

    # =========================================================
    def process(self, frame):

        self.frame_count += 1

        # ================= FRAME SKIP =================
        if self.frame_count % self.frame_skip != 0:
            return False, frame

        frame = cv2.resize(frame, (640, 480))

        # ================= DETECTION =================
        results = self.model(frame, conf=0.5, classes=[0], verbose=False)

        boxes = results[0].boxes
        detected = boxes is not None and len(boxes) > 0

        # ================= RECORDING =================
        if detected:
            self.no_detection_frames = 0

            if not self.recording:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = f"event_{timestamp_str}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(video_name, fourcc, 20, (640, 480))
                self.recording = True

            # write frame
            if self.out is not None:
                self.out.write(frame)

        else:
            if self.recording:
                self.no_detection_frames += 1

                if self.no_detection_frames > self.stop_threshold:
                    self.recording = False
                    self.out.release()
                    self.out = None

        # ================= CSV LOG =================
        if detected:
            for _ in boxes:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                self.csv_writer.writerow([
                    self.serial_no,
                    "emergency_vehicle",
                    timestamp
                ])

                self.serial_no += 1

        return detected, frame

    # =========================================================
    def release(self):
        self.csv_file.close()
        if self.out is not None:
            self.out.release()