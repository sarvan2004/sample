import cv2
import numpy as np
from ultralytics import YOLO

from Crash import CrashDetector
from Emergency import EmergencyDetector
from nearmiss import NearMissDetector
from pedistrian import PedestrianViolationDetector

from PedestrianCounter import PedestrianCounter
from SpeedDetector import SpeedDetector


# ================= ROI DRAW (POLYGON) =================
def draw_polygon(frame, window_name):
    points = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse)

    while True:
        temp = frame.copy()

        for p in points:
            cv2.circle(temp, p, 5, (0, 255, 0), -1)

        if len(points) >= 2:
            cv2.polylines(temp, [np.array(points)], False, (0, 255, 255), 2)

        if len(points) >= 3:
            cv2.polylines(temp, [np.array(points)], True, (0, 255, 255), 2)

        cv2.putText(temp, "Click points | S=Save | R=Reset",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow(window_name, temp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points.clear()

        elif key == ord('s') and len(points) >= 3:
            break

    cv2.destroyWindow(window_name)
    return points


# ================= LINE DRAW =================
def draw_line(frame, window_name):
    pts = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse)

    while True:
        temp = frame.copy()

        for p in pts:
            cv2.circle(temp, p, 5, (0, 255, 0), -1)

        if len(pts) == 2:
            cv2.line(temp, pts[0], pts[1], (0, 255, 0), 2)

        cv2.putText(temp, "Click 2 points | S=Save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow(window_name, temp)

        if cv2.waitKey(1) & 0xFF == ord('s') and len(pts) == 2:
            break

    cv2.destroyWindow(window_name)
    return pts


# ================= MAIN =================
def main():

    VIDEO_PATH = "Emergency_sample.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read video")
        return

    print("🔄 Loading models...")
    shared_model = YOLO("yolov8n.pt")

    # ================= ROI SETUP =================
    print("\n🎯 Draw Crash ROI")
    crash_roi = draw_polygon(first_frame, "Crash ROI")

    print("\n⚠️ Draw Near Miss ROI")
    near_roi = draw_polygon(first_frame, "Near Miss ROI")

    # ================= PEDESTRIAN ROIs =================
    print("\n🚶 Draw Pedestrian ROI A")
    ped_roi_A = draw_polygon(first_frame, "Ped ROI A")

    print("\n🚶 Draw Pedestrian ROI B")
    ped_roi_B = draw_polygon(first_frame, "Ped ROI B")

    print("\n🚶 Draw Pedestrian ROI C")
    ped_roi_C = draw_polygon(first_frame, "Ped ROI C")

    print("\n🚶 Draw Pedestrian ROI D")
    ped_roi_D = draw_polygon(first_frame, "Ped ROI D")

    ped_rois = [ped_roi_A, ped_roi_B, ped_roi_C, ped_roi_D]

    # ================= SPEED LINES (A–F) =================
    print("\n🚗 Draw Speed Lines (A–F)")

    print("Lane 1 → Draw A then B")
    lineA = draw_line(first_frame, "Line A")
    lineB = draw_line(first_frame, "Line B")

    print("Lane 2 → Draw C then D")
    lineC = draw_line(first_frame, "Line C")
    lineD = draw_line(first_frame, "Line D")

    print("Lane 3 → Draw E then F")
    lineE = draw_line(first_frame, "Line E")
    lineF = draw_line(first_frame, "Line F")

    speed_lines = {
        "AB": (lineA, lineB),
        "CD": (lineC, lineD),
        "EF": (lineE, lineF)
    }

    # ================= INIT MODULES =================
    crash = CrashDetector(model=shared_model, roi_points=crash_roi)
    emergency = EmergencyDetector(model_path="best.pt")
    near = NearMissDetector(model=shared_model, roi_points=near_roi)
    ped = PedestrianViolationDetector(model=shared_model, roi_points=ped_rois)

    ped_count = PedestrianCounter(model=shared_model, roi_points=ped_rois)
    speed = SpeedDetector(model=shared_model, line_pairs=speed_lines)

    print("\n🚀 System Running...\n")

    # ================= LOOP =================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crash_flag, frame = crash.process(frame)
        emergency_flag, frame = emergency.process(frame)
        near_flag, frame = near.process(frame)
        ped_flag, frame = ped.process(frame)

        counts, frame = ped_count.process(frame)
        speed_flag, frame = speed.process(frame)

        # ================= DISPLAY =================
        y = 30

        if crash_flag:
            cv2.putText(frame, "CRASH", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            y += 25

        if emergency_flag:
            cv2.putText(frame, "EMERGENCY", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y += 25

        if near_flag:
            cv2.putText(frame, "NEAR MISS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            y += 25

        if ped_flag:
            cv2.putText(frame, "PEDESTRIAN VIOLATION", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            y += 25

        if speed_flag:
            cv2.putText(frame, "SPEED", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            y += 25

        # ================= PEDESTRIAN COUNT =================
        for i, c in enumerate(counts):
            cv2.putText(frame, f"Ped Count {i}: {c}",
                        (300, 30 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Unified AI System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ================= CLEANUP =================
    cap.release()
    cv2.destroyAllWindows()

    crash.release()
    emergency.release()
    near.release()
    ped.release()
    ped_count.release()
    speed.release()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
