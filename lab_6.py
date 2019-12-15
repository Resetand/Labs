import cv2
import numpy as np

from lib.opecv_utils import fetch_image
from lib.print import bcolors

# Лабораторная работа 6
# Тема: Детектирование и отслеживание на видео
# Задание:
# 1. Разработать программу детектирования движения на видео с использованием так
# называемой истории движения (motionHistoryImage()) (пример на рис).

# 2. Метод Лукаса-Канаде. Визуализировать разреженный оптический поток с помощью
# функции calcOpticalFlowPyrLK ()

# 3. Применить любой из трекеров отслеживания, реализованных в библиотеке OpenCV к видео.
# 4. Реализовать алгоритм YOLO: https://pjreddie.com/darknet/yolo/

cv2.waitKey(0)
cv2.destroyAllWindows()
