from typing import Any
import torch
import numpy as np
import cv2 
from time import time
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision import BoxAnnotator
from supervision.annotators.utils import Detections

humanDetectionInstances = 0
threshold = 10 # Avoid sending help signal on false positives

class ObjectDetection:
    def __init__(self, capture_index):

        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: " + self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=3, text_thickness=2)

    def load_model(self):
        model = YOLO("yolov8m.pt")
        model.fuse()

        return model
    
    def predict(self, frame):
        results = self.model.predict(frame)
        return results
    
    def plot_boxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extracts detections
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0: # only extracts persons
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        global humanDetectionInstances
        global threshold

        print(xyxys)

        if np.size(xyxys) > 0:
            humanDetectionInstances += 1

        if humanDetectionInstances > threshold: # human detected, send help
            print("Human detected. Sending help...")

        # Formats and sets up detections for visualization using supervision
        detections = Detections(
            xyxy = results[0].boxes.xyxy.cpu().numpy(),
            confidence = results[0].boxes.conf.cpu().numpy(),
            class_id = results[0].boxes.cls.cpu().numpy().astype(int),
        )

        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, _, confidence, class_id, tracker_id
                       in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(frame, detections=detections, labels=self.labels)
        return frame
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        capture = cv2.VideoCapture(self.capture_index)
        assert capture.isOpened()
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 630*2)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 340*2)

        while True: 
            start_time = time()
            isTrue, frame = capture.read()

            assert isTrue

            results = self.predict(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF==27: # Esc
                break
        
        capture.release()
        cv2.destroyAllWindows()


# Testing the code
#detector = ObjectDetection(capture_index='https://10.0.0.10:8080/video') # selecting webcam
detector = ObjectDetection(capture_index=0)
detector()