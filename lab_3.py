import cv2
import numpy as np
import argparse
from lib.utils import Utils
from lib.helpers import ImageURL


class ShapsesBasic:
    def get_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(src=gray, thresh=127, maxval=255, type=0)
        return cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]

    def contours_process(self):
        image = Utils.fetch_image(url=ImageURL.RANDOM)
        orig = image.copy()
        contours = self.get_contours(image)

        cv2.drawContours(
            image=image, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=3)

        cv2.putText(img=image,
                    text=f"contours: {str(len(contours))}", org=(10, 40),
                    fontScale=1.2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 0), thickness=2)
        return image, orig

    def get_hough_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return cv2.HoughLines(edges, 1, np.pi/180, 200)

    def hough_lines_process(self):

        image = Utils.fetch_image(url=ImageURL.BUILDING)
        orig = image.copy()

        lines = self.get_hough_lines(image)

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image, orig

    def get_hough_circles(self, img):
        img = cv2.medianBlur(img, 7)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        return circles

    def hough_circles_process(self):
        image = Utils.fetch_image(url=ImageURL.OPENCV_LOGO)
        orig = image.copy()
        circles = self.get_hough_circles(image)

        for [x, y, raduis] in circles[0, :]:
            cv2.circle(img=image,
                       center=(x, y), radius=raduis,
                       color=(0, 0, 255), thickness=2)

        return image, orig


def process():
    parser = argparse.ArgumentParser()

    parser.add_argument('--contours', '-co', action='store_true')
    parser.add_argument('--circle', '-ci', action='store_true')
    parser.add_argument('--lines', '-l', action='store_true', default=True)

    args = parser.parse_args()

    shapsesBasic = ShapsesBasic()
    orig = None
    result = None

    if args.contours:
        result, orig = shapsesBasic.contours_process()

    elif args.circle:
        result, orig = shapsesBasic.hough_circles_process()

    elif args.lines:
        result, orig = shapsesBasic.hough_lines_process()

    Utils.show_image_compare(orig, result)


process()
