import cv2

from lib.utils import Utils


mat = Utils.fetch_image(url='https://source.unsplash.com/random/200x400',
                        flag=cv2.IMREAD_GRAYSCALE)

cv2.imshow('GRAYSCALE image', mat)

cv2.waitKey(0)
