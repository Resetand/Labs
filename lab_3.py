import cv2
import numpy as np

from lib.opecv_utils import fetch_image
from lib.print import bcolors

# Лабораторная работа 3
# Тема: Работа с контурами, выделение линий и кругов
# 	•	Подсчитайте контуры на изображении используя функцию поиска контуров
#       cvFindContours(). За исходное изображение возьмите изображение
#       однотипных предметов на контрастном фоне (например, монетки, кубики и т.д.)
# 	•	Найдите прямые линии на изображении при помощи преобразований Хофа,
#       функция cvHoughLines2(). В качестве исходного изображения лучше использовать
#       фотографию дородной разметки, дома, улицы.
# 	•	Найдите окружности на изображении при помощи преобразований Хофа, функция cvHoughCircles ().


def count_and_draw_contours():
    # Ищем контрасное фото
    image = fetch_image(
        url='https://source.unsplash.com/random/400x800?logo')

    # Делаем чб для threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Функция threshold возвращает изображение,
    # в котором все пиксели, которые темнее (меньше) 127
    # заменены на 0, а все, которые ярче (больше) 127, — на 255.
    # таким образом мы делаем фото еще более контрастным
    _, thresh = cv2.threshold(src=gray, thresh=127, maxval=255, type=0)

    # Находим контуры как массив массивов [х, y] координат
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    print(f"\n{bcolors.OKGREEN}Number of contours = {bcolors.OKBLUE}{len(contours)} {bcolors.ENDC}")

    # Рисуем сразу все контуры
    # Копируем img для избежания мутирования массива
    # Передаем контуры
    # contourIdx = -1 говорит нарисовать нам все контуры
    with_contours = cv2.drawContours(
        image=image.copy(), contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
    cv2.imshow('@count_and_draw_contours / original image', image)
    cv2.imshow('@count_and_draw_contours / image with contours', with_contours)


def draw_lines():
    """
        Преобразование Хафа— это метод для поиска линий, кругов и других простых форм на изображении.
    """
    url = "https://images.unsplash.com/photo-1487653557405-97ba52327f93?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&h=800&ixid=eyJhcHBfaWQiOjF9&ixlib=rb-1.2.1&q=80&w=400"

    image = fetch_image(url=url)

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

    cv2.imshow('@draw_lines / hough lines3', image)


def draw_circle():

    url = "https://i.pinimg.com/originals/d5/14/77/d5147798dc96186eb172a41ffbbeab78.jpg"

    image = fetch_image(url=url)

    image_copy = image.copy()

    # Заблюрим лишние детали
    image = cv2.medianBlur(image, 7)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Резульатом будует массив массивов вида
    # [[x y raduis-lenght], где x, y - координтаты центра
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    # Округлим значения
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(image_copy, (i[0], i[1]), i[2], (0, 0, 255), 2)

    cv2.imshow('@draw_circle / result', image_copy)


count_and_draw_contours()
draw_lines()
draw_circle()

cv2.waitKey(0)
cv2.destroyAllWindows()
