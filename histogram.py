import cv2 as cv
import numpy as np

def get_accumulative_hist(image, mask):
    height, width = image.shape[0], image.shape[1]
    hist = cv.calcHist([image], [0], mask, [256], [0, 256])
    ratio_hist = hist / (height * width)
    accumulative_hist = np.zeros(256, np.float64)
    ratio_sum = 0
    for i in range(256):
        ratio_sum += ratio_hist[i]
        accumulative_hist[i] = ratio_sum

    return accumulative_hist


def hist_match_channel_one(src, ref, mask):
    mapped_img = np.zeros(src.shape, src.dtype)

    src_accumulative_hist = get_accumulative_hist(src, mask)
    ref_accumulative_hist = get_accumulative_hist(ref, mask)

    map_array_c1 = np.zeros(256, src.dtype)
    for i, src_hist_bin in enumerate(src_accumulative_hist):
        min_diff_abs = 1000
        min_diff_abs_index = 0
        for j, ref_hist_bin in enumerate(ref_accumulative_hist):
            diff_abs = np.abs(ref_hist_bin - src_hist_bin)

            if diff_abs < min_diff_abs:
                min_diff_abs = diff_abs
                min_diff_abs_index = j

        map_array_c1[i] = min_diff_abs_index

    src_height = src.shape[0]
    src_width = src.shape[1]

    for row in range(src_height):
        for col in range(src_width):
            src_color = src[row, col]
            if mask is not None and src[row, col] != 0:
                map_color = map_array_c1[src_color]
                mapped_img[row, col] = map_color
            else:
                mapped_img[row, col] = src_color

    return mapped_img


def Histogram_Matching(src_img, ref_img):
    img1_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    _, mask1 = cv.threshold(src=img1_gray, thresh=7, maxval=255, type=cv.THRESH_BINARY)
    _, mask2 = cv.threshold(src=img2_gray, thresh=7, maxval=255, type=cv.THRESH_BINARY)
    mask = mask1 & mask2
    src_channels = cv.split(src_img)
    ref_channels = cv.split(ref_img)

    src_mapped_channels = []
    for i in range(3):
        src_mapped_channel = hist_match_channel_one(src_channels[i], ref_channels[i], mask)
        src_mapped_channels.append(src_mapped_channel)

    src_mapped = cv.merge(src_mapped_channels)
    return src_mapped