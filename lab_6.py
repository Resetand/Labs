
from yolov.index import YolovService
from lib.utils import Utils


def process():
    Utils.capture_webcam(
        (lambda frame: YolovService(frame).detect_process()))


process()
