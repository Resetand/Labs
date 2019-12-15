import cv2
import os


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

face_dacase = cv2.CascadeClassifier(f'{dir_path}/faces.xml')
eyes_dacase = cv2.CascadeClassifier(f'{dir_path}/eyes.xml')
body_dacase = cv2.CascadeClassifier(f'{dir_path}/bodies.xml')


class Recognition:

    def __init__(self):
        self.faces = 0
        self.eyes = 0
        self.bodies = 0

    def logger(self):
        print(f"{bcolors.OKGREEN} Number of:  faces = {self.faces} {bcolors.OKBLUE} | eyes = {self.eyes}  {bcolors.HEADER} | humans = {self.bodies} {bcolors.ENDC} ")

    def bodyDetect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bodies = body_dacase.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in bodies:
            cv2.rectangle(
                img=img,
                pt1=(x, y),
                pt2=(x+w, y+h),
                color=(0, 0, 255),
                thickness=3
            )

        self.bodies = len(bodies)
        return bodies

    def eyesDetect(self, img, faces):
        eyes_count = 0
        eyes_global = []
        for (fx, fy, fw, fh) in faces:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray_area = gray[fy:fy+fh, fx:fx+fw]
            color_area = img[fy:fy+fh, fx:fx+fw]

            eyes = eyes_dacase.detectMultiScale(gray_area)
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

    def faceDetect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_dacase.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img=img,
                pt1=(x, y),
                pt2=(x+w, y+h),
                color=(0, 0, 255),
                thickness=2
            )
        self.faces = len(faces)
        return faces

    def startCaptureWebcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()

            faces = self.faceDetect(img)
            self.eyesDetect(img, faces)
            self.bodyDetect(img)

            self.logger()
            cv2.imshow('recognition', img)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()


SERVICE = Recognition()
SERVICE.startCaptureWebcam()
