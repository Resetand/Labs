import cv2
import argparse
from lib.utils import Utils


class ProcessingBasic:
    """
        Лабораторная работа №2
        Тема: «Вывод видео. Детектирование краев»
        Разработать приложение, выводящее видео с камеры.
        Примените к видео в реальном времени:

        •	оператор Собеля
        •	оператор Лапласа
        •	детектор границ Кенни.
    """

    def __init__(self, img):
        self.img = img

    def sobel(self):
        """
            дискретный дифференциальный оператор, 
            вычисляющий приближение градиента яркости изображения.
            Оператор вычисляет градиент яркости изображения в каждой точке. 
            Так находится направление наибольшего увеличения яркости и величина её 
            изменения в этом направлении. Результат показывает, насколько «резко» или 
            «плавно» меняется яркость изображения в каждой точке, а значит, вероятность
             нахождения точки на грани, а также ориентацию границы.

            Т.о. результатом работы оператора Собеля в точке области постоянной 
            яркости будет нулевой вектор, а в точке, лежащей на границе областей 
            различной яркости — вектор, пересекающий границу в направлении увеличения яркости.

            Наиболее часто оператор Собеля применяется в алгоритмах выделения границ.
            3 <= ksize <= 31  && ksize is odd!!!
        """
        sobelx = cv2.Sobel(self.img.copy(), cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.img.copy(), cv2.CV_64F, 0, 1, ksize=5)
        return sobelx

    def canny(self):
        """
            Canny является методом выделения границ.
            Работает так
            — Убрать шум и лишние детали из изображения
            — Рассчитать градиент изображения
            — Сделать края тонкими (edge thinning)
            — Связать края в контура (edge linking)
        """
        canny = cv2.Canny(self.img.copy(), 100, 200)
        return canny

    def laplacian(self):
        """
            суммирование производных второго порядка
            фактически, это оператор собеля с dx = dy = 2
        """
        laplacian = cv2.Laplacian(self.img.copy(), cv2.CV_64F)
        return laplacian


def process():
    parser = argparse.ArgumentParser(description='Process webcam stream.')

    parser.add_argument('--laplacian', '-l', action='store_true')
    parser.add_argument('--sobel', '-s', action='store_true')
    parser.add_argument('--canny', '-c', action='store_true', default=True)

    args = parser.parse_args()

    if args.laplacian:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).laplacian(), 'laplacian')
    elif args.sobel:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).sobel(), 'sobel')
    elif args.canny:
        Utils.capture_webcam(
            lambda frame: ProcessingBasic(frame).canny(), 'canny')


process()
