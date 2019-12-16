import cv2
from lib.utils import Utils
from lib.helpers import ImageURL

cv2.imshow('GRAYSCALE image', Utils.fetch_image(
    url=ImageURL.RANDOM, flag=cv2.IMREAD_GRAYSCALE))
cv2.waitKey(0)
