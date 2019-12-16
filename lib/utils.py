from urllib.request import urlopen
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Utils:

    @staticmethod
    def fetch_image(url, flag=cv2.IMREAD_COLOR):
        req = urlopen(url)
        image = np.asarray(bytearray(req.read()), dtype="uint8")
        return cv2.imdecode(image, flag)

    @staticmethod
    def capture_webcam(img_handler=(lambda frame: frame), winname='Open-cv'):
        """ @param img_hwandler (Mat) => Mat: """
        cap = cv2.VideoCapture(0)
        while (True):
            _, frame = cap.read()
            frame = img_handler(frame)
            pressed_key = cv2.waitKey(5) & 0xFF
            cv2.imshow(winname, frame)
            if pressed_key == 27:
                break
        cv2.destroyAllWindows()
        cap.release()

    @staticmethod
    def show_image_compare(orig, result, title=''):
        figure = plt.figure()
        figure.canvas.set_window_title(title)
        figure.add_subplot(1, 2, 1)

        plt.imshow(orig)
        figure.add_subplot(1, 2, 2)
        plt.imshow(result)
        plt.show(block=True)
