import cv2

from lib.opecv_utils import fetch_image
from lib.utils import show_image_compare


class SpecificAreas:
    """
        Лабораторная работа №4
        Тема: Определение характерных точек. Аффинные преобразования.

        •	Выделить характерные угловые точки на произвольном изображении при
            помощи детектора углов Харриса (cornerHarris()).

        •	Выделить характерные угловые точки при помощи детектора углов Ши Томаси
            (goodFeaturesToTrack ()).

        •	Воспользуйтесь функцией аффинных преобразований getAffineTransform()
            или функцией перспективных преобразований getPerspectiveTransform.()
            для поворота изображения (например, для последующего распознавания). Для этого
            задания в качестве исходного изображения необходимо использовать фото лежащего на
            столе листа с текстом или книги, т.е. изначально изображение должно быть искажено.
    """

    def bootstrap(self):
        corner_harris, corner_harris_orig = self.draw_corner_harris()
        show_image_compare(corner_harris_orig, corner_harris, 'Corner Harris')

        good_features_to_track, good_features_to_track_orig = self.draw_good_features_to_track()
        show_image_compare(good_features_to_track_orig,
                           good_features_to_track, 'Corner Harris')

    def draw_good_features_to_track(self):
        img = fetch_image(
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
        img = fetch_image(
            'https://source.unsplash.com/random/400x800?chess')
        orig = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dts = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=.0001)

        img[dts > 0.01 * dts.max()] = [0, 0, 255]
        return img, orig


SpecificAreas().bootstrap()
