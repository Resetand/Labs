
from recognition.index import Recognition
from lib.utils import Utils


def process():
    Utils.capture_webcam((lambda frame: Recognition(frame).detect_all()))


process()
