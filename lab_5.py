
from recognition.index import RecognitionService
from lib.utils import Utils


def process():
    Utils.capture_webcam(
        (lambda frame: RecognitionService(frame).detect_process()))


process()
