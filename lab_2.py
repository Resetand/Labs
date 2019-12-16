import cv2
import argparse
from lib.utils import Utils


class ProcessingBasic:
    def __init__(self, img):
        self.img = img

    def sobel_process(self):
        sobelx = cv2.Sobel(self.img.copy(), cv2.CV_64F, 1, 0, ksize=5)
        return sobelx

    def canny_process(self):
        canny = cv2.Canny(self.img.copy(), 100, 200)
        return canny

    def laplacian_process(self):
        laplacian = cv2.Laplacian(self.img.copy(), cv2.CV_64F)
        return laplacian


def process():
    parser = argparse.ArgumentParser()

    parser.add_argument('--laplacian', '-l', action='store_true')
    parser.add_argument('--sobel', '-s', action='store_true')
    parser.add_argument('--canny', '-c', action='store_true', default=True)

    args = parser.parse_args()

    if args.laplacian:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).laplacian_process(), 'laplacian')
    elif args.sobel:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).sobel_process(), 'sobel')
    elif args.canny:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).canny_process(), 'canny')


process()
