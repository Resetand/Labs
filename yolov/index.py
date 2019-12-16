import cv2
import numpy as np
import os

dir_path = os.path.dirname(__file__)


net = cv2.dnn.readNet(f"{dir_path}/data/yolov3.weights",
                      f"{dir_path}/data/yolov3.cfg")


with open(f"{dir_path}/data/coco.names", 'r') as f:
    CLASSES = [line.strip() for line in f]


LAYER_NAMES = net.getLayerNames()
OUTPUT_LAYERS = [LAYER_NAMES[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class Detection:
    def __init__(self, detection, img):
        self.img = img
        self.detection = detection
        (center_x, center_y, width, height) = self.getShape()
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.class_id = np.argmax(detection[5:])
        self.confidence = self.getConfidence()

        self.label = CLASSES[self.class_id]

        self.box = [int(center_x - (width / 2)),
                    int(center_y - (height / 2)), int(width), int(height)]

    def getConfidence(self):
        scores = self.detection[5:]
        class_id = np.argmax(scores)
        return scores[class_id]   # 0 --> 1

    def getShape(self):
        (img_height, img_width) = self.img.shape[:2]
        box = self.detection[0:4] * \
            np.array([img_width, img_height, img_width, img_height])
        (center_x, center_y, width, height) = box.astype("int")
        return (center_x, center_y, width, height)


class DetectionService:
    def __init__(self, img):
        self.img = img.copy()
        self.blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.detections = self.detect_all()

    def detect_all(self):
        net.setInput(self.blob)
        outs = net.forward(OUTPUT_LAYERS)
        detections = []
        for out in outs:
            for detect in out:
                detection = Detection(detect, self.img)
                if detection.confidence > .5:
                    detections.append(detection)
        return detections

    def draw_all(self):
        for detection in self.detections:
            self.draw_detect(detection)
        return self.img

    def draw_detect(self, detection):
        [x, y, w, h] = detection.box
        text = f"{detection.label}: {round(detection.confidence*100)}%"
        cv2.circle(self.img, (detection.center_x,
                              detection.center_y), 1, (0, 0, 255), 1)
        cv2.putText(self.img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1)
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
