import cv2
from lib.utils import Utils
from lib.helpers import ImageURL

img =  Utils.fetch_image(url=ImageURL.RANDOM)
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Utils.show_image_compare(img, gray)
cv2.waitKey(0)
