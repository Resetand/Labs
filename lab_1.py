import cv2

from lib.opecv_utils import fetch_image


mat = fetch_image(url='https://source.unsplash.com/random/200x400',
                  flag=cv2.IMREAD_GRAYSCALE)

cv2.imshow('GRAYSCALE image', mat)

cv2.waitKey(0)
