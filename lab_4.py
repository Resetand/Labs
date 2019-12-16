import cv2
import argparse
from lib.utils import Utils
from matplotlib import pyplot as plt


class SpecificAreas:
    """
        Лабораторная работа №4
        Тема: Определение характерных точек. Аффинные преобразования.

        •	Выделить характерные угловые точки на произвольном изображении при
            помощи детектора углов Харриса (cornerHarris()).

        •	Выделить характерные угловые точки при помощи детектора углов Ши Томаси
            (goodFeaturesToTrack ()).
    """

    def draw_good_features_to_track(self):
        img = Utils.fetch_image(
            'https://source.unsplash.com/random/400x800?chess')
        orig = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        koseler = cv2.goodFeaturesToTrack(
            image=gray, maxCorners=50, qualityLevel=0.0001, minDistance=10)

        for kose in koseler:
            x, y = kose.ravel()
            cv2.circle(img=img, center=(x, y), radius=3,
                       color=(0, 0, 255), thickness=-1)

        return img, orig

    def draw_corner_harris(self):
        img = Utils.fetch_image(
            'https://source.unsplash.com/random/400x800?chess')
        orig = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dts = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=.0001)

        img[dts > 0.01 * dts.max()] = [0, 0, 255]
        return img, orig


def process():
    parser = argparse.ArgumentParser()

    parser.add_argument('--harris', '-ha', action='store_true', default=True)
    parser.add_argument('--koseler', '-k', action='store_true')

    args = parser.parse_args()

    specificAreas = SpecificAreas()
    orig = None
    result = None

    if args.koseler:
        result, orig = specificAreas.draw_good_features_to_track()

    elif args.harris:
        result, orig = specificAreas.draw_corner_harris()

    Utils.show_image_compare(orig, result)
    plt.waitforbuttonpress()


process()
