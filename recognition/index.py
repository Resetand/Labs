import cv2
import os
from clear_screen import clear


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Лабораторная работа 5
# Тема: Использование каскада Хаара для классификации изображений
# Задание:
# Разработать программу обнаружения на видео лица, глаз и фигуры человека в полный рост,
# используя каскады Хаара.
# Для работы понадобятся готовые обученные классификаторы XML,
# которые можно найти на официальной странице OpenCV на GitHub.

dir_path = os.path.dirname(__file__)

FACE_DACASE = cv2.CascadeClassifier(f'{dir_path}/faces.xml')
EYES_DACASE = cv2.CascadeClassifier(f'{dir_path}/eyes.xml')
BODY_DACASE = cv2.CascadeClassifier(f'{dir_path}/bodies.xml')


class Recognition:

    def __init__(self, img):
        self.img = img
        self.faces = 0
        self.eyes = 0
        self.bodies = 0

    def logger(self):
        clear()
        print(f"\n\n\nStatistic:  {bcolors.OKGREEN} faces = {self.faces} {bcolors.OKBLUE} \t\t | \t\t eyes = {self.eyes}  {bcolors.HEADER} \t\t|\t\t humans = {self.bodies} {bcolors.ENDC} ")

    def body_detect(self):

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        bodies = BODY_DACASE .detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in bodies:
            cv2.rectangle(
                img=self.img,
                pt1=(x, y),
                pt2=(x+w, y+h),
                color=(0, 0, 255),
                thickness=3
            )

        self.bodies = len(bodies)
        return bodies

    def eyes_detect(self, faces):
        eyes_count = 0
        eyes_global = []
        for (fx, fy, fw, fh) in faces:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            gray_area = gray[fy:fy+fh, fx:fx+fw]
            color_area = self.img[fy:fy+fh, fx:fx+fw]

            eyes = EYES_DACASE.detectMultiScale(gray_area)
            eyes_global += [eyes]
            eyes_count += len(eyes)
            for (x, y, w, h) in eyes:
                cv2.rectangle(
                    img=color_area,
                    pt1=(x, y),
                    pt2=(x+w, y+h),
                    color=(0, 255, 0),
                    thickness=1
                )

        self.eyes = eyes_count

        return eyes_global

    def face_detect(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = FACE_DACASE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img=self.img,
                pt1=(x, y),
                pt2=(x+w, y+h),
                color=(0, 0, 255),
                thickness=2
            )
        self.faces = len(faces)
        return faces

    def detect_all(self):
        faces = self.face_detect()
        self.eyes_detect(faces)
        self.body_detect()
        self.logger()
        return self.img
