import cv2
from itertools import product

import numpy as np


def median_cor_coeff_spp(block_size, image):
    h, w = image.shape[:2]
    y, x = h // 2 - block_size//2, w // 2 - block_size//2
    img_patch = img[y:y+block_size, x:x+block_size]

    result = cv2.matchTemplate(image, img_patch, cv2.TM_CCOEFF_NORMED)
    return np.median(result)


if __name__ == "__main__":
    img = cv2.imread("./t166.png", cv2.IMREAD_COLOR)

   #print(cv2.matchTemplate(img, img[:-2, :-2], cv2.TM_CCORR_NORMED))
   #a = np.ones_like(img)*255
   #print(cv2.matchTemplate(a, a[:-2, :-2], cv2.TM_CCOEFF))
   #quit()

    interval = min(img.shape[:2]) // 2
    lb, ub = 3, round(min(img.shape[:2])/2.5)
    g_best_bs = 0
    g_best_v  = -float("inf")
    g_worst_bs = 0
    g_worst_v  = +float("inf")

    # SEARCH BEST
    while interval != 1:
        interval = interval // 8
        if interval == 0:
            interval = 1
        print(f"interval : {interval}")

        best_v = -float("inf")
        best_bs = g_best_bs

        for block_size in range(ub, lb, -interval):
            if block_size == g_best_bs:
                continue

            print(f"searching... bs: {block_size}")
            med_cc_sums = median_cor_coeff_spp(block_size, img)
            if med_cc_sums > best_v:
                best_v = med_cc_sums
                best_bs = block_size

        if best_v > g_best_v:
            g_best_v = best_v
            g_best_bs = best_bs

        lb = max(lb, g_best_bs - interval)
        ub = min(ub, g_best_bs + interval)
        print(f"best :  {(best_bs, best_v)}")


    # search WORST
    print("______________________________")
    interval = min(img.shape[:2]) // 2
    lb, ub = 3, round(min(img.shape[:2]) / 2.5)
    while interval != 1:
        interval = interval // 8
        if interval == 0:
            interval = 1
        print(f"interval : {interval}")

        worst_v = float("inf")
        worst_bs= 0

        for block_size in range(ub, lb, -interval):
            if block_size == g_worst_bs:
                continue

            print(f"searching... bs: {block_size}")
            med_cc_sums = median_cor_coeff_spp(block_size, img)
            if med_cc_sums < worst_v:
                worst_v = med_cc_sums
                worst_bs= block_size

        if worst_v < g_worst_v:
            g_worst_v = worst_v
            g_worst_bs = worst_bs

        lb = max(lb, g_worst_bs - interval)
        ub = min(ub, g_worst_bs + interval)

    print(f"best :  {(g_best_bs, g_best_v)} ")
    print(f"worst:  {(g_worst_bs, g_worst_v)} ")
    #print(f"mcc:  {median_cor_coeff_spp(33, img)} ")




