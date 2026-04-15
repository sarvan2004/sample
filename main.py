import cv2
import numpy as np
from ultralytics import YOLO

from Crash import CrashDetector
from Emergency import EmergencyDetector
from NearMiss import NearMissDetector
from pedistrian import PedestrianViolationDetector


# ================= ROI DRAW =================
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


# ================= MAIN =================
def main():

    VIDEO_PATH = "Hour-1.mp4"
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

    # ================= PEDESTRIAN MULTI ROI =================
    print("\n🚶 Draw Pedestrian ROI A")
    ped_roi_A = draw_polygon(first_frame, "Ped ROI A")

    print("\n🚶 Draw Pedestrian ROI B")
    ped_roi_B = draw_polygon(first_frame, "Ped ROI B")

    print("\n🚶 Draw Pedestrian ROI C")
    ped_roi_C = draw_polygon(first_frame, "Ped ROI C")

    print("\n🚶 Draw Pedestrian ROI D")
    ped_roi_D = draw_polygon(first_frame, "Ped ROI D")

    ped_rois = [ped_roi_A, ped_roi_B, ped_roi_C, ped_roi_D]

    # ================= INIT MODULES =================
    crash = CrashDetector(model=shared_model, roi_points=crash_roi)

    emergency = EmergencyDetector(model_path="best.pt")  # ✅ no ROI

    near = NearMissDetector(model=shared_model, roi_points=near_roi)

    ped = PedestrianViolationDetector(model=shared_model, roi_points=ped_rois)

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
            cv2.putText(frame, "PEDESTRIAN", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Unified AI System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    crash.release()
    emergency.release()
    near.release()
    ped.release()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()