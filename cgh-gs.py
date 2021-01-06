# -*- coding: utf-8 -*-

import cv2


target = cv2.imread("img/target.bmp", 0)
cv2.imshow("target", target)
cv2.waitKey(0)
