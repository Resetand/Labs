
from yolov.index import DetectionService
from lib.utils import Utils


def process():
    Utils.capture_webcam((lambda frame: DetectionService(frame).draw_all()))


process()
