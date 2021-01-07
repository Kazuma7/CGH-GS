# -*- coding: utf-8 -*-
import cv2
import numpy as np


#RMSE評価関数
def root_mean_squared_error(pred, true):
    hr = 150
    wr = 150                                       
    pred_area = pred[int(height/2-hr):int(height/2+hr),int(width/2-wr):int(width/2+wr)]
    true_area = true[int(height/2-hr):int(height/2+hr),int(width/2-wr):int(width/2+wr)]
    mse = np.power(true_area - pred_area, 2).mean()
    return np.sqrt(mse)

#多点に対する光の強度の均一性
def check_uniformity(u_int, target):
    u_int = u_int / np.max(u_int)
    maxi = np.max(u_int[target == 1])
    mani = np.min(u_int[target == 1])
    uniformity = 1 - (maxi - mini) / (maxi + mini)
    return uniformity

#正規化
def normalization(origin):
    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm

#ホログラム画像生成
def hologram(phase):
    holophase = np.where(phase < 0, phase + 2 * np.pi, phase)
    #条件分岐 np.where(condition x y)
    p_max = np.max(phase)
    p_min = np.min(phase)
    holo = ((phase - p_min) / (p_max - p_min)) * 255
    holo = holo.astype("uint8")

#再生像生成
def reconstruct(norm_int):
    rec = norm_int * 255
    rec = rec.astype("uint8")
    return rec




target = cv2.imread("img/target1.bmp", 0)
# cv2.imshow("target", target)
# cv2.waitKey(0)

height, width = target.shape[:2]

target = target / 255
laser = 1
phase = np.random.rand(height, width)
u = np.empty_like(target, dtype="complex")
# dtypeは実部と虚部が存在する空行列

iteration = 5

#CGHの複素振幅の作成
u.real = laser * np.cos(phase)
u.imag = laser * np.sin(phase)

for num in range(iteration):

    #フーリエ変換
    u = np.fft.fft2(u)
    u = np.fft.fftshift(u)

    u_abs = np.abs(u)
    u_int = u_abs ** 2
    norm_int = normalization(u_int)

    #再生像の元
    norm_int = norm

    rec = reconstruct(norm_int)

    phase = np.angle(u)

    #再生像の複素振幅をターゲットに合わせて修正
    u.real = target * np.cos(phase)
    u.imag = target * np.sin(phase)

    #逆フーリエ変換
    u = np.fft.ifftshift(u)
    u = np.fft.ifft2(u)

    #CGHの元を生成
    phase = np.angle(u)

    holo = hologram(phase)

    holo_name = "holo"
    rec_name = "rec"

    filename_holo = "img/holo" + str(num+1) + ".png"
    filename_rec = "img/rec" + str(num+1) + ".png"  
    
    cv2.imwrite(filename_holo, holo)
    # cv2.imshow("Hologram", holo)
    # cv2.waitKey(0)

    cv2.imwrite(filename_rec, rec)
    # cv2.imshow("Reconstruction", rec)
    # cv2.waitKey(0)

    mse = root_mean_squared_error(norm_int, target)
    print(mse)
    print(check_uniformity(u_int, target))






