# -*- coding: utf-8 -*-

import cv2
import numpy as np

target = cv2.imread("img/target.bmp", 0)
# cv2.imshow("target", target)
# cv2.waitKey(0)

height, width = target.shape[:2]

print(height)
print(width)

# print(target)

target = target / 255
laser = 1
phase = np.random.rand(height, width)
u = np.empty_like(target, dtype="complex")
# dtypeは実部と虚部が存在する空行列

print(phase)
# print(u)

iteration = 20

for num in range(iteration):
    
    u.real = laser * np.cos(phase)
    u.imag = laser * np.sin(phase)

    # print(np.cos(phase[1,1]))
    # print(np.sin(phase[1,1]))

    # print(u[1, 1])

    u = np.fft.fft2(u)
    u = np.fft.fftshift(u)

    # print(u[1,1])

    u_abs = np.abs(u)
    u_int = u_abs ** 2

    maxi = np.max(u_int)
    print('最大:',maxi)
    mini = np.min(u_int)
    print('最小:',mini)

    norm = ((u_int - mini) / (maxi - mini))

    norm_int = norm

    # print(norm_int)

    phase = np.angle(u)
    print(phase)

    u.real = target * np.cos(phase)
    u.imag = target * np.sin(phase)

    u = np.fft.ifftshift(u)
    u = np.fft.ifft2(u)

    print('IFFT後')
    phase = np.angle(u)
    print(phase)

    holophase = np.where(phase < 0, phase + 2 * np.pi, phase)
    p_max = np.max(phase)
    p_min = np.min(phase)
    holo = ((phase - p_min) / (p_max - p_min)) * 255
    holo = holo.astype("uint8")

    rec = norm_int * 255
    rec = rec.astype("uint8")

    holo_name = "holo"
    rec_name = "rec"

    if num == 0:
    
        cv2.imwrite("img/{}.bmp".format(holo_name), holo)
        cv2.imshow("Hologram", holo)
        cv2.waitKey(0)

        cv2.imwrite("img/{}.bmp".format(rec_name), rec)
        cv2.imshow("Reconstruction", rec)
        cv2.waitKey(0)

    elif num == 19:
        cv2.imwrite("img/{}1.bmp".format(holo_name), holo)
        cv2.imshow("Hologram", holo)
        cv2.waitKey(0)

        cv2.imwrite("img/{}1.bmp".format(rec_name), rec)
        cv2.imshow("Reconstruction", rec)
        cv2.waitKey(0)

    print(num)





