import cv2
import argparse
from lib.utils import Utils
from lib.helpers import ImageURL
from matplotlib import pyplot as plt
import numpy as np


class SpecificAreas:

    def draw_good_features_to_track(self):
        img = Utils.fetch_image(ImageURL.CHESS)
        orig = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        koseler = cv2.goodFeaturesToTrack(
            image=gray, maxCorners=50, qualityLevel=0.0001, minDistance=10)

        for kose in koseler:
            x, y = kose.ravel()
            cv2.circle(img=img, center=(x, y), radius=3,
                       color=(0, 0, 255), thickness=-1)

        return img, orig

    def draw_corner_harris(self):
        img = Utils.fetch_image(ImageURL.CHESS)
        orig = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dts = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=.0001)

        img[dts > 0.01 * dts.max()] = [0, 0, 255]
        return img, orig

    def affin_rotate(self):
        src = Utils.fetch_image(ImageURL.RANDOM)
        orig = src.copy()
        srcTri = np.array([[0, 0], [src.shape[1] - 1, 0],
                           [0, src.shape[0] - 1]]).astype(np.float32)
        dstTri = np.array([[0, src.shape[1]*0.33], [src.shape[1]*0.85, src.shape[0]
                                                    * 0.25], [src.shape[1]*0.15, src.shape[0]*0.7]]).astype(np.float32)

        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

        center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)

        angle = 90
        scale = 1
        rotated = cv2.getRotationMatrix2D(center, angle, scale)
        return src, orig


def process():
    parser = argparse.ArgumentParser()

    parser.add_argument('--harris', '-ha', action='store_true', default=True)
    parser.add_argument('--koseler', '-k', action='store_true')
    parser.add_argument('--affine', '-a', action='store_true')

    args = parser.parse_args()

    specificAreas = SpecificAreas()
    orig = None
    result = None

    if args.affine:
        result, orig = specificAreas.affin_rotate()

    elif args.koseler:
        result, orig = specificAreas.draw_good_features_to_track()

    elif args.harris:
        result, orig = specificAreas.draw_corner_harris()

    Utils.show_image_compare(orig, result)
    plt.waitforbuttonpress()


process()
