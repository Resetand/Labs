from urllib.request import urlopen
import cv2
import numpy as np


def fetch_image(url, flag=cv2.IMREAD_COLOR):
    """
        Fetch image by url and return the cv2 mat object
    """
    req = urlopen(url)
    image = np.asarray(bytearray(req.read()), dtype="uint8")
    return cv2.imdecode(image, flag)
