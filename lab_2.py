import cv2


class CaptureService:
    """
        Лабораторная работа №2
        Тема: «Вывод видео. Детектирование краев»
        Разработать приложение, выводящее видео с камеры.
        Примените к видео в реальном времени:

        •	оператор Собеля
        •	оператор Лапласа
        •	детектор границ Кенни.
    """

    def __init__(self):
        self.img = None

    def start_capture(self):
        cap = cv2.VideoCapture(0)
        while (True):
            _, frame = cap.read()
            self.img = frame

            self.show_sobel()
            self.show_canny()
            self.show_laplacian()

            pressed_key = cv2.waitKey(5) & 0xFF
            if pressed_key == 27:
                break
        cv2.destroyAllWindows()
        cap.release()

    def show_sobel(self):
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
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)

    def show_canny(self):
        """
            Canny является методом выделения границ.
            Работает так
            — Убрать шум и лишние детали из изображения
            — Рассчитать градиент изображения
            — Сделать края тонкими (edge thinning)
            — Связать края в контура (edge linking)
        """
        canny = cv2.Canny(self.img.copy(), 100, 200)
        cv2.imshow('Canny', canny)

    def show_laplacian(self):
        """
            суммирование производных второго порядка
            фактически, это оператор собеля с dx = dy = 2
        """
        laplacian = cv2.Laplacian(self.img.copy(), cv2.CV_64F)
        cv2.imshow('Laplacian', laplacian)


CaptureService().start_capture()
