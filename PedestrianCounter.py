import cv2
import numpy as np
from ultralytics import YOLO

PEDESTRIAN_CLASS = 0
VEHICLE_CLASSES = [2, 3, 5, 7]

class GeometryEngine:
    def __init__(self, polygon):
        self.crosswalk_poly = np.array(polygon, dtype=np.int32)

    def bbox_in_crosswalk(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cv2.pointPolygonTest(self.crosswalk_poly, (cx, cy), False) >= 0


class PedestrianCounter:
    def __init__(self, model, roi_points):
        self.model = model
        self.geometries = [GeometryEngine(r) for r in roi_points]

    def process(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)

        counts = [0] * len(self.geometries)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                cls = int(box.cls[0])
                bbox = tuple(box.xyxy[0].cpu().numpy())

                if cls == PEDESTRIAN_CLASS:
                    for i, geo in enumerate(self.geometries):
                        if geo.bbox_in_crosswalk(bbox):
                            counts[i] += 1

        return counts, frame

    def release(self):
        pass