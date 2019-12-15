import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.opecv_utils import fetch_image
from lib.utils import show_image_compare


class Shapses:
    """
        Лабораторная работа 3
        Тема: Работа с контурами, выделение линий и кругов

        •	Подсчитайте контуры на изображении используя функцию поиска контуров
            cvFindContours(). За исходное изображение возьмите изображение
            однотипных предметов на контрастном фоне (например, монетки, кубики и т.д.)

        •	Найдите прямые линии на изображении при помощи преобразований Хофа,
            функция cvHoughLines2(). В качестве исходного изображения лучше использовать
            фотографию дородной разметки, дома, улицы.

        •	Найдите окружности на изображении при помощи преобразований Хофа, функция cvHoughCircles().
    """

    def bootstrap(self):
        contours, orig_contours = self.count_and_draw_contours()
        show_image_compare(orig_contours, contours, "contours")

        lines, orig_lines = self.draw_lines()
        show_image_compare(orig_lines, lines, "Hough lines")

        circles, orig_circles = self.draw_circle()
        show_image_compare(orig_circles, circles, "Hough circles")

        plt.waitforbuttonpress()

    def count_and_draw_contours(self):
        """
            Сначала ищем контрасное изображение на unsplash
            Копируем его 
            Дополнительно делаем холст с чб изображением и пропускаем 
            его через threshold
            ( Функция threshold возвращает изображение,
            в котором все пиксели, которые темнее (меньше) 127
            заменены на 0, а все, которые ярче (больше) 127, — на 255.
            таким образом мы делаем фото еще более контрастным )

            По этому контрасному холсту и считаем контуры 

            Находим контуры как массив вот таких[х, y] координат

            Добавляем инфу о кол-ве контуров прямо в img
            И рисуем сразу все контуры (contourIdx = -1 говорит нарисовать все)
        """

        image = fetch_image(
            url='https://source.unsplash.com/random/400x800?logo')

        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(src=gray, thresh=127, maxval=255, type=0)

        contours, _ = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        cv2.putText(img=image,
                    text=f"contours: {str(len(contours))}", org=(10, 40),
                    fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 255, 0), lineType=1)
        cv2.drawContours(
            image=image, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
        return image, orig

    def draw_lines(self):
        """
            Преобразование Хафа— это метод для поиска линий,
            кругов и других простых форм на изображении
        """

        url = "https://images.unsplash.com/photo-1487653557405-97ba52327f93?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&h=800&ixid=eyJhcHBfaWQiOjF9&ixlib=rb-1.2.1&q=80&w=400"

        image = fetch_image(url=url)
        orig = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

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

    def draw_circle(self):
        """
            Преобразование Хафа— это метод для поиска линий,
            кругов и других простых форм на изображении

        """
        url = "https://i.pinimg.com/originals/d5/14/77/d5147798dc96186eb172a41ffbbeab78.jpg"

        image = fetch_image(url=url)
        orig = image.copy()

        image = cv2.medianBlur(image, 7)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))

        for [x, y, raduis] in circles[0, :]:
            cv2.circle(img=image,
                       center=(x, y), radius=raduis,
                       color=(0, 0, 255), thickness=2)

        return image, orig


Shapses().bootstrap()
