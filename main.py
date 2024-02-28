import cv2
import maxflow
from energy import get_energy_map
import numpy as np
import matchers
from ransac import RANSAC
from homography import homography_fit, get_hom_final_size
from imagewarping import imagewarping, imagewarping1, imagewarping2
from apap import APAP_stitching, get_mdlt_final_size
import config
import histogram
import random
import immatch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import Ssim
import Psnr


def input1(img1, img2, mask, mask1):
    ## SIFT keypoint detection and matching
    matcher_obj = matchers.matchers()
    kp1, ds1 = matcher_obj.getFeatures(img1, mask)
    kp2, ds2 = matcher_obj.getFeatures(img2, mask1)
    matches = matcher_obj.match(ds1, ds2)
    src_orig = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_orig = np.float32([kp2[m.trainIdx].pt for m in matches])
    src_orig1 = np.copy(src_orig)
    dst_orig1 = np.copy(dst_orig)
    src_orig = np.vstack((src_orig.T, np.ones((1, len(matches)))))
    dst_orig = np.vstack((dst_orig.T, np.ones((1, len(matches)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(img1, img2, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    RMSE = []
    MSE = []
    for k in range(len(src_orig1)):
        x, y = dst_orig1[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(src_orig1[k] - point1)
        distance_2 = distance ** 2
        RMSE.append(distance_2)
        MSE.append(distance)
    mse = np.mean(MSE)
    rmse = np.sqrt(np.mean(RMSE))
    min_x, max_x, min_y, max_y = get_hom_final_size(img1, img2, Hg)
    linear_hom, com_image, same = imagewarping1(img1, img2, Hg, min_x, max_x, min_y, max_y)
    X, Y = np.meshgrid(np.linspace(0, img2.shape[1]-1, config.C1), np.linspace(0, img2.shape[0]-1, config.C2))
    Mv = np.array([X.ravel(), Y.ravel()]).T
    apap = APAP_stitching(config.gamma, config.sigma)
    Hmdlt = apap(dst_fine, src_fine, Mv)
    min_x, max_x, min_y, max_y = get_mdlt_final_size(img1, img2, Hmdlt, config.C1, config.C2)
    warped_img1 = imagewarping2(img1, img2, Hmdlt, min_x, max_x, min_y, max_y, config.C1, config.C2)
    return linear_hom, Hg, com_image, same, warped_img1, rmse, mse

def input2(img1, img2):
    cv2.imwrite("result5/3.jpg", img1)
    cv2.imwrite("result5/4.jpg", img2)
    with open('image-matching-toolbox/configs/superglue.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['example']
    model = immatch.__dict__[args['class']](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    matches, _, _, _ = matcher("result5/3.jpg", "result5/4.jpg") 
    p0 = np.array([[10, 20]], dtype=np.float32)
    p1 = np.array([[10, 20]], dtype=np.float32)
    for i in range(len(matches)):
        x1 = matches[i, 0]
        y1 = matches[i, 1]
        x2 = matches[i, 2]
        y2 = matches[i, 3] 
        new_feature = np.array([[x1, y1]], dtype=np.float32)
        p0 = np.append(p0, new_feature, axis=0)
        new_feature = np.array([[x2, y2]], dtype=np.float32)
        p1 = np.append(p1, new_feature, axis=0)
    p0 = p0[1:]
    p1 = p1[1:]
    src_orig = np.vstack((p0.T, np.ones((1, len(p0)))))
    dst_orig = np.vstack((p1.T, np.ones((1, len(p1)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    RMSE = []
    MSE = []
    for k in range(len(p0)):
        x, y = p1[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(p0[k] - point1)
        distance_2 = distance ** 2
        RMSE.append(distance_2)
        MSE.append(distance)
    mse = np.mean(MSE)
    rmse = np.sqrt(np.mean(RMSE))
    min_x, max_x, min_y, max_y = get_hom_final_size(img1, img2, Hg)
    linear_hom, com_image, same = imagewarping1(img1, img2, Hg, min_x, max_x, min_y, max_y)
    return linear_hom, Hg, com_image, same, rmse, mse

"""
def input2(img1, img2, map_x, map_y, off):
    warped_img2 = cv2.remap(img2, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask2 = np.ones((img2.shape[0], img2.shape[1]))
    warped_mask2 = cv2.remap(mask2, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped_img3 = np.zeros_like(warped_img2)
    warped_img4 = np.zeros_like(warped_img2)
    warped_img4[off[1]:off[1] + img1.shape[0], off[0]:off[0] + img1.shape[1], :] = img1
    warped_img5 = np.zeros_like(warped_img2)
    for c in range(3):
        warped_img3[:, :, c] = warped_img2[:, :, c] * warped_mask2
        warped_img5[:, :, c] = warped_img4[:, :, c] * warped_mask2
    warped_img6 = warped_img3[off[1]:off[1] + img1.shape[0], off[0]:off[0] + img1.shape[1]]
    warped_img7 = warped_img5[off[1]:off[1] + img1.shape[0], off[0]:off[0] + img1.shape[1]]
    warped_img8 = histogram.Histogram_Matching(warped_img6, warped_img7)
    return warped_img8
"""

def seamcut1(img1, img2):
    src = img1
    dst = img2

    img_pixel1, img_pixel2, left, right, up, down = get_energy_map(src, dst)

    g = maxflow.GraphFloat()
    img_pixel1 = img_pixel1.astype(float)
    img_pixel1 = img_pixel1 * 1e10
    img_pixel2 = img_pixel2.astype(float)
    img_pixel2 = img_pixel2 * 1e10
    nodeids = g.add_grid_nodes(img_pixel1.shape)

    g.add_grid_tedges(nodeids, img_pixel1, img_pixel2)
    structure_left = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [0, 0, 0]])
    g.add_grid_edges(nodeids, weights=left, structure=structure_left, symmetric=False)
    structure_right = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 0, 0]])
    g.add_grid_edges(nodeids, weights=right, structure=structure_right, symmetric=False)
    structure_up = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 1, 0]])
    g.add_grid_edges(nodeids, weights=up, structure=structure_up, symmetric=False)
    structure_down = np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
    g.add_grid_edges(nodeids, weights=down, structure=structure_down, symmetric=False)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)

    img2 = np.int_(np.logical_not(sgm))
    src_mask = img2.astype(np.uint8)
    dst_mask = np.logical_not(img2).astype(np.uint8)
    src_mask = np.stack((src_mask, src_mask, src_mask), axis=-1)
    dst_mask = np.stack((dst_mask, dst_mask, dst_mask), axis=-1)
    maskImg = np.zeros(src_mask.shape[:2], dtype=float)
    maskImg[dst_mask[:, :, 0] > 0] = 1.0

    src = src * src_mask
    dst = dst * dst_mask

    result = src + dst
    return result, src_mask, dst_mask

def seamcut2(img1, img2, src_mask, dst_mask):
    src = img1
    dst = img2
    src = src * src_mask
    dst = dst * dst_mask

    result = src + dst
    return result


def map1(frame, frame1, mask, mask1, file_name_without_extension, image_index):
    result, Hg1, com_image, same_loc, apap, rmse1, mse1 = input1(frame, frame1, mask, mask1)
    cv2.imwrite("result5/1.jpg", result)
    cv2.imwrite("result5/2.jpg", same_loc)
    with open('image-matching-toolbox/configs/superglue.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['example']
    model = immatch.__dict__[args['class']](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    matches, _, _, _ = matcher("result5/1.jpg", "result5/2.jpg") 
    p0 = np.array([[10, 20]], dtype=np.float32)
    p1 = np.array([[10, 20]], dtype=np.float32)
    for i in range(len(matches)):
        x1 = matches[i, 0]
        y1 = matches[i, 1]
        x2 = matches[i, 2]
        y2 = matches[i, 3] 
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        com_image = cv2.line(com_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        new_feature = np.array([[x1, y1]], dtype=np.float32)
        p0 = np.append(p0, new_feature, axis=0)
        new_feature = np.array([[x2, y2]], dtype=np.float32)
        p1 = np.append(p1, new_feature, axis=0)
    file_name = f"result5/{file_name_without_extension}_{image_index}.jpg"
    cv2.imwrite(file_name, com_image)
    p0 = p0[1:]
    p1 = p1[1:]
    src_orig = np.vstack((p1.T, np.ones((1, len(p1)))))
    dst_orig = np.vstack((p0.T, np.ones((1, len(p0)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    RMSE = []
    MSE = []
    for k in range(len(p0)):
        x, y = p0[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(p1[k] - point1)
        distance_2 = distance ** 2
        RMSE.append(distance_2)
        MSE.append(distance)
    mse = np.mean(MSE)
    rmse = np.sqrt(np.mean(RMSE))

    min_x, max_x, min_y, max_y = get_hom_final_size(frame, frame, Hg)
    linear_hom = imagewarping(frame, result, Hg, min_x, max_x, min_y, max_y)
    file_name = f"result7/{file_name_without_extension}_{image_index}.jpg"
    cv2.imwrite(file_name, linear_hom)
    return linear_hom, Hg, result, apap, rmse, mse, rmse1, mse1
"""
def map11(frame, frame1, mask, mask1, file_name_without_extension, image_index):
    result, Hg1, com_image = input1(frame, frame1, mask, mask1)
    com_image1 = np.copy(com_image)
    Hg1 = np.linalg.inv(Hg1)
    result1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 5))
    h, w, s = frame.shape
    p0 = np.array([[10, 20]], dtype=np.float32)
    p2 = np.array([[10, 20]], dtype=np.float32)
    p3 = np.array([[10, 20]], dtype=np.float32)
    for i in range(96):
        for j in range(36):
            img = frame1[j * h // 36:(1 + j) * h // 36, i * w // 96:(1 + i) * w // 96, ]
            kp1 = cv2.FastFeatureDetector_create().detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            if (len(kp1) <= 4):
                for k in range(len(kp1)):
                    if i >= 48 or j >= 18:
                        x, y = kp1[k].pt
                        x = x + i * w // 96
                        y = y + j * h // 36
                        source_point = np.array([x, y, 1])
                        mapped_point = np.dot(Hg1, source_point)
                        mapped_point_normalized = mapped_point / mapped_point[2]
                        mapped_x, mapped_y = mapped_point_normalized[:2]
                        x1, y1 = int(mapped_x), int(mapped_y)
                        new_feature = np.array([[x1, y1]], dtype=np.float32)
                        p0 = np.append(p0, new_feature, axis=0)
            else:
                selected = random.sample(kp1, 4)
                for k in range(len(selected)):
                    if i >= 48 or j >= 18:
                        x, y = kp1[k].pt
                        x = x + i * w // 96
                        y = y + j * h // 36
                        source_point = np.array([x, y, 1])
                        mapped_point = np.dot(Hg1, source_point)
                        mapped_point_normalized = mapped_point / mapped_point[2]
                        mapped_x, mapped_y = mapped_point_normalized[:2]
                        x1, y1 = int(mapped_x), int(mapped_y)
                        new_feature = np.array([[x1, y1]], dtype=np.float32)
                        p0 = np.append(p0, new_feature, axis=0)
    p0 = p0[1:]
    p1, st, err = cv2.calcOpticalFlowPyrLK(result1, frame_gray, p0, None, **lk_params)
    src_orig = np.vstack((p1.T, np.ones((1, len(p1)))))
    dst_orig = np.vstack((p0.T, np.ones((1, len(p0)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    Distance = []
    for k in range(len(p0)):
        x, y = p0[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(p1[k] - point1)
        Distance.append(distance)
    distance_low = np.sort(Distance)[0]
    distance_high = np.sort(Distance)[-800]
    distance_th = 0.5 * distance_low + 0.5 * distance_high
    err_low = np.sort(err)[0]
    err_high = np.sort(err)[-800]
    err_th = 0.4 * err_low + 0.6 * err_high
    for k in range(len(p0)):
        if Distance[k] <= distance_th and st[k] == 1 and err[k] <= err_th:
            x, y = p0[k]
            new_feature = np.array([[x, y]], dtype=np.float32)
            p2 = np.append(p2, new_feature, axis=0)
            x, y = p1[k]
            new_feature = np.array([[x, y]], dtype=np.float32)
            p3 = np.append(p3, new_feature, axis=0)
    p2 = p2[1:]
    p3 = p3[1:]
    for k in range(len(p2)):
        a, b = p2[k]
        c, d = p3[k]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        com_image1 = cv2.line(com_image1, (int(a), int(b)), (int(c), int(d)), color, 2)
    file_name = f"result4/{file_name_without_extension}_{image_index}.jpg"
    cv2.imwrite(file_name, com_image1)
    src_orig = np.vstack((p3.T, np.ones((1, len(p3)))))
    dst_orig = np.vstack((p2.T, np.ones((1, len(p2)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    min_x, max_x, min_y, max_y = get_hom_final_size(frame, frame, Hg)
    linear_hom = imagewarping(frame, result, Hg, min_x, max_x, min_y, max_y)
    return linear_hom, Hg

def map21(frame, frame1, hg, mask, mask1, file_name_without_extension, image_index):
    result, Hg1, com_image = input1(frame, frame1, mask, mask1)
    com_image1 = np.copy(com_image)
    Hg1 = np.linalg.inv(Hg1)
    result1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 5))
    h, w, s = frame.shape
    p0 = np.array([[10, 20]], dtype=np.float32)
    p2 = np.array([[10, 20]], dtype=np.float32)
    p3 = np.array([[10, 20]], dtype=np.float32)
    for i in range(96):
        for j in range(36):
            img = frame1[j * h // 36:(1 + j) * h // 36, i * w // 96:(1 + i) * w // 96, ]
            kp1 = cv2.FastFeatureDetector_create().detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            if (len(kp1) <= 4):
                for k in range(len(kp1)):
                    if i >= 48 or j >= 18:
                        x, y = kp1[k].pt
                        x = x + i * w // 96
                        y = y + j * h // 36
                        source_point = np.array([x, y, 1])
                        mapped_point = np.dot(Hg1, source_point)
                        mapped_point_normalized = mapped_point / mapped_point[2]
                        mapped_x, mapped_y = mapped_point_normalized[:2]
                        x1, y1 = int(mapped_x), int(mapped_y)
                        new_feature = np.array([[x1, y1]], dtype=np.float32)
                        p0 = np.append(p0, new_feature, axis=0)
            else:
                selected = random.sample(kp1, 4)
                for k in range(len(selected)):
                    if i >= 48 or j >= 18:
                        x, y = kp1[k].pt
                        x = x + i * w // 96
                        y = y + j * h // 36
                        source_point = np.array([x, y, 1])
                        mapped_point = np.dot(Hg1, source_point)
                        mapped_point_normalized = mapped_point / mapped_point[2]
                        mapped_x, mapped_y = mapped_point_normalized[:2]
                        x1, y1 = int(mapped_x), int(mapped_y)
                        new_feature = np.array([[x1, y1]], dtype=np.float32)
                        p0 = np.append(p0, new_feature, axis=0)
    p0 = p0[1:]
    p1, st, err = cv2.calcOpticalFlowPyrLK(result1, frame_gray, p0, None, **lk_params)
    src_orig = np.vstack((p1.T, np.ones((1, len(p1)))))
    dst_orig = np.vstack((p0.T, np.ones((1, len(p0)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    Distance = []
    for k in range(len(p0)):
        x, y = p0[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(p1[k] - point1)
        Distance.append(distance)
    distance_low = np.sort(Distance)[0]
    distance_high = np.sort(Distance)[-800]
    distance_th = 0.5 * distance_low + 0.5 * distance_high
    err_low = np.sort(err)[0]
    err_high = np.sort(err)[-500]
    err_th = 0.4 * err_low + 0.6 * err_high
    for k in range(len(p0)):
        if Distance[k] <= distance_th and st[k] == 1:
            x, y = p0[k]
            new_feature = np.array([[x, y]], dtype=np.float32)
            p2 = np.append(p2, new_feature, axis=0)
            x, y = p1[k]
            new_feature = np.array([[x, y]], dtype=np.float32)
            p3 = np.append(p3, new_feature, axis=0)
    p2 = p2[1:]
    p3 = p3[1:]
    for k in range(len(p2)):
        a, b = p2[k]
        c, d = p3[k]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        com_image1 = cv2.line(com_image1, (int(a), int(b)), (int(c), int(d)), color, 2)
    file_name = f"result4/{file_name_without_extension}_{image_index}.jpg"
    cv2.imwrite(file_name, com_image1)
    src_orig = np.vstack((p3.T, np.ones((1, len(p3)))))
    dst_orig = np.vstack((p2.T, np.ones((1, len(p2)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    eigenvalues, eigenvectors = np.linalg.eig(hg)
    D = np.diag(eigenvalues ** 0.5)
    B = eigenvectors
    A_pow_0_4 = B @ D @ np.linalg.inv(B)
    eigenvalues, eigenvectors = np.linalg.eig(Hg)
    D = np.diag(eigenvalues ** 0.5)
    B = eigenvectors
    B_pow_0_6 = B @ D @ np.linalg.inv(B)
    A = np.dot(A_pow_0_4, B_pow_0_6)
    min_x, max_x, min_y, max_y = get_hom_final_size(frame, frame, A)
    linear_hom = imagewarping(frame, result, A, min_x, max_x, min_y, max_y)
    return linear_hom
"""
def map2(frame, frame1, hg, mask, mask1, file_name_without_extension, image_index):
    result, Hg1, com_image, same_loc, apap, rmse1, mse1 = input1(frame, frame1, mask, mask1)
    cv2.imwrite("result5/1.jpg", result)
    cv2.imwrite("result5/2.jpg", same_loc)
    with open('image-matching-toolbox/configs/superglue.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['example']
    model = immatch.__dict__[args['class']](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    matches, _, _, _ = matcher("result5/1.jpg", "result5/2.jpg") 
    p0 = np.array([[10, 20]], dtype=np.float32)
    p1 = np.array([[10, 20]], dtype=np.float32)
    for i in range(len(matches)):
        x1 = matches[i, 0]
        y1 = matches[i, 1]
        x2 = matches[i, 2]
        y2 = matches[i, 3] 
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        com_image = cv2.line(com_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        new_feature = np.array([[x1, y1]], dtype=np.float32)
        p0 = np.append(p0, new_feature, axis=0)
        new_feature = np.array([[x2, y2]], dtype=np.float32)
        p1 = np.append(p1, new_feature, axis=0)
    file_name = f"result5/{file_name_without_extension}_{image_index}.jpg"
    cv2.imwrite(file_name, com_image)
    p0 = p0[1:]
    p1 = p1[1:]
    src_orig = np.vstack((p1.T, np.ones((1, len(p1)))))
    dst_orig = np.vstack((p0.T, np.ones((1, len(p0)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    """
    p2 = np.array([[10, 20]], dtype=np.float32)
    p3 = np.array([[10, 20]], dtype=np.float32)
    for k in range(len(p0)):
        x, y = p0[k]
        source_point = np.array([x, y, 1])
        mapped_point = np.dot(np.linalg.inv(Hg), source_point)
        mapped_point_normalized = mapped_point / mapped_point[2]
        mapped_x, mapped_y = mapped_point_normalized[:2]
        point1 = np.array([mapped_x, mapped_y])
        distance = np.linalg.norm(p1[k] - point1)
        if distance <= 10:
            new_feature = np.array([[x, y]], dtype=np.float32)
            p2 = np.append(p2, new_feature, axis=0)
            x, y = p1[k]
            new_feature = np.array([[x, y]], dtype=np.float32)
            p3 = np.append(p3, new_feature, axis=0)
    p2 = p2[1:]
    p3 = p3[1:]
    for k in range(len(p2)):
        a, b = p2[k]
        c, d = p3[k]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        com_image1 = cv2.line(com_image1, (int(a), int(b)), (int(c), int(d)), color, 2)
    file_name = f"result5/{file_name_without_extension}_{image_index}_copy.jpg"        
    cv2.imwrite(file_name, com_image1)
    src_orig = np.vstack((p3.T, np.ones((1, len(p3)))))
    dst_orig = np.vstack((p2.T, np.ones((1, len(p2)))))
    ransac = RANSAC(config.M, config.thr)
    src_fine, dst_fine = ransac(frame, frame, src_orig, dst_orig)
    Hg = homography_fit(src_fine, dst_fine)
    """
    eigenvalues, eigenvectors = np.linalg.eig(hg)
    D = np.diag(eigenvalues ** 0.5)
    B = eigenvectors
    A_pow_0_4 = B @ D @ np.linalg.inv(B)
    eigenvalues, eigenvectors = np.linalg.eig(Hg)
    D = np.diag(eigenvalues ** 0.5)
    B = eigenvectors
    B_pow_0_6 = B @ D @ np.linalg.inv(B)
    A = np.dot(A_pow_0_4, B_pow_0_6)
    min_x, max_x, min_y, max_y = get_hom_final_size(frame, frame, A)
    linear_hom = imagewarping(frame, result, A, min_x, max_x, min_y, max_y)
    return linear_hom, result, apap

if __name__ == '__main__':
    video = cv2.VideoCapture('test4/videos/中间1.mp4')
    video1 = cv2.VideoCapture('test4/videos/右下1.mp4')
    video2 = cv2.VideoCapture('test4/videos/右上1.mp4')
    video3 = cv2.VideoCapture('test4/videos/左下1.mp4')
    video4 = cv2.VideoCapture('test4/videos/左上1.mp4')
    Hg1, Hg2, Hg3, Hg4 = None, None, None, None
    src_mask, dst_mask = None, None
    src_mask1, dst_mask1 = None, None
    src_mask2, dst_mask2 = None, None
    ret, frame = video.read()
    print(frame.shape)
    height, width, _ = frame.shape
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("result5/19.mp4", fourcc, 25, (width, height))
    video_writer1 = cv2.VideoWriter("result5/21.mp4", fourcc, 25, (width, height))
    video_writer2 = cv2.VideoWriter("result5/22.mp4", fourcc, 25, (width, height))
    video_writer3 = cv2.VideoWriter("result5/23.mp4", fourcc, 25, (width, height))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 25)
    frame_count = 0
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask1 = np.zeros_like(gray_image, dtype=np.uint8)
    mask2 = np.zeros_like(gray_image, dtype=np.uint8)
    mask3 = np.zeros_like(gray_image, dtype=np.uint8)
    mask4 = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (1920, 1080), 255, -1)
    cv2.rectangle(mask1, (960, 540), (1920, 1080), 255, -1)
    cv2.rectangle(mask2, (960, 0), (1920, 540), 255, -1)
    cv2.rectangle(mask3, (0, 540), (960, 1080), 255, -1)
    cv2.rectangle(mask4, (0, 0), (960, 540), 255, -1)
    """
    fast_right_bottom_rmse = []
    normal_right_bottom_rmse = []
    super_right_bottom_rmse = []
    fast_right_bottom_mse = []
    normal_right_bottom_mse = []
    super_right_bottom_mse = []
    fast_right_top_rmse = []
    normal_right_top_rmse = []
    super_right_top_rmse = []
    fast_right_top_mse = []
    normal_right_top_mse = []
    super_right_top_mse = []
    fast_left_bottom_rmse = []
    normal_left_bottom_rmse = []
    super_left_bottom_rmse = []
    fast_left_bottom_mse = []
    normal_left_bottom_mse = []
    super_left_bottom_mse = []
    fast_left_top_rmse = []
    normal_left_top_rmse = []
    super_left_top_rmse = []
    fast_left_top_mse = []
    normal_left_top_mse = []
    super_left_top_mse = []
    """
    while True:
        ret, frame = video.read()
        ret, frame1 = video1.read()
        ret, frame2 = video2.read()
        ret, frame3 = video3.read()
        ret, frame4 = video4.read()
        if i >= 525:
            break
        if frame_count % frame_interval == 0 and i == 400:
            """
            (b, g, r) = cv2.split(frame)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
            bh = clahe.apply(b)
            gh = clahe.apply(g)
            rh = clahe.apply(r)
            frame = cv2.merge((bh, gh, rh), )
            """
            result_fast, Hg1, result_normal, result_apap, rmse, mse, rmse1, mse1 = map1(frame, frame1, mask1, mask, "right_bottom", i)
            #fast_right_bottom_rmse.append(rmse)
            #normal_right_bottom_rmse.append(rmse1)
            #fast_right_bottom_mse.append(mse)
            #normal_right_bottom_mse.append(mse1)
            result_fast1, Hg2, result_normal1, result_apap1, rmse, mse, rmse1, mse1 = map1(frame, frame2, mask2, mask, "right_top", i)
            #fast_right_top_rmse.append(rmse)
            #normal_right_top_rmse.append(rmse1)
            #fast_right_top_mse.append(mse)
            #normal_right_top_mse.append(mse1)
            result_fast2, Hg3, result_normal2, result_apap2, rmse, mse, rmse1, mse1 = map1(frame, frame3, mask3, mask, "left_bottom", i)
            #fast_left_bottom_rmse.append(rmse)
            #normal_left_bottom_rmse.append(rmse1)
            #fast_left_bottom_mse.append(mse)
            #normal_left_bottom_mse.append(mse1)
            result_fast3, Hg4, result_normal3, result_apap3, rmse, mse, rmse1, mse1 = map1(frame, frame4, mask4, mask, "left_top", i)
            #fast_left_top_rmse.append(rmse)
            #normal_left_top_rmse.append(rmse1)
            #fast_left_top_mse.append(mse)
            #normal_left_top_mse.append(mse1)

            """
            result_super, hg, com_image, same, rmse, mse = input2(frame, frame1)
            #super_right_bottom_rmse.append(rmse)
            #super_right_bottom_mse.append(mse)
            result_super1, hg, com_image, same, rmse, mse = input2(frame, frame2)
            #super_right_top_rmse.append(rmse)
            #super_right_top_mse.append(mse)
            result_super2, hg, com_image, same, rmse, mse = input2(frame, frame3)
            #super_left_bottom_rmse.append(rmse)
            #super_left_bottom_mse.append(mse)
            result_super3, hg, com_image, same, rmse, mse = input2(frame, frame4)
            #super_left_top_rmse.append(rmse)
            #super_left_top_mse.append(mse)
            """
            """
            result_fast1 = histogram.Histogram_Matching(result_fast1, result_fast)
            result_fast2 = histogram.Histogram_Matching(result_fast2, result_fast)
            result_fast3 = histogram.Histogram_Matching(result_fast3, result_fast2)
            """
            """
            result_fast2 = histogram.Histogram_Matching(result_fast2, result_fast3)
            result_fast1 = histogram.Histogram_Matching(result_fast1, result_fast3)
            result_fast = histogram.Histogram_Matching(result_fast, result_fast1)

            result_normal2 = histogram.Histogram_Matching(result_normal2, result_normal3)
            result_normal1 = histogram.Histogram_Matching(result_normal1, result_normal3)
            result_normal = histogram.Histogram_Matching(result_normal, result_normal1)

            result_apap2 = histogram.Histogram_Matching(result_apap2, result_apap3)
            result_apap1 = histogram.Histogram_Matching(result_apap1, result_apap3)
            result_apap = histogram.Histogram_Matching(result_apap, result_apap1)

            result_super2 = histogram.Histogram_Matching(result_super2, result_super3)
            result_super1 = histogram.Histogram_Matching(result_super1, result_super3)
            result_super = histogram.Histogram_Matching(result_super, result_super1)
            """
            #数据集1的直方图方法
            result_fast = histogram.Histogram_Matching(result_fast, frame)
            result_fast1 = histogram.Histogram_Matching(result_fast1, frame)
            result_fast2 = histogram.Histogram_Matching(result_fast2, frame)
            result_fast3 = histogram.Histogram_Matching(result_fast3, frame)
            """

            result_normal = histogram.Histogram_Matching(result_normal, frame)
            result_normal1 = histogram.Histogram_Matching(result_normal1, frame)
            result_normal2 = histogram.Histogram_Matching(result_normal2, frame)
            result_normal3 = histogram.Histogram_Matching(result_normal3, frame)

            result_apap = histogram.Histogram_Matching(result_apap, frame)
            result_apap1 = histogram.Histogram_Matching(result_apap1, frame)
            result_apap2 = histogram.Histogram_Matching(result_apap2, frame)
            result_apap3 = histogram.Histogram_Matching(result_apap3, frame)

            result_super = histogram.Histogram_Matching(result_super, frame)
            result_super1 = histogram.Histogram_Matching(result_super1, frame)
            result_super2 = histogram.Histogram_Matching(result_super2, frame)
            result_super3 = histogram.Histogram_Matching(result_super3, frame)
            """

            cut_result_fast, src_mask1, dst_mask1 = seamcut1(result_fast, result_fast1)
            cut_result_fast1, src_mask2, dst_mask2 = seamcut1(result_fast2, result_fast3)
            final_result_fast, src_mask, dst_mask = seamcut1(cut_result_fast, cut_result_fast1)
            """
            cut_result_normal = seamcut2(result_normal, result_normal1, src_mask1, dst_mask1)
            cut_result_normal1 = seamcut2(result_normal2, result_normal3, src_mask2, dst_mask2)
            final_result_normal = seamcut2(cut_result_normal, cut_result_normal1, src_mask, dst_mask)

            cut_result_apap = seamcut2(result_apap, result_apap1, src_mask1, dst_mask1)
            cut_result_apap1 = seamcut2(result_apap2, result_apap3, src_mask2, dst_mask2)
            final_result_apap = seamcut2(cut_result_apap, cut_result_apap1, src_mask, dst_mask)

            cut_result_super = seamcut2(result_super, result_super1, src_mask1, dst_mask1)
            cut_result_super1 = seamcut2(result_super2, result_super3, src_mask2, dst_mask2)
            final_result_super = seamcut2(cut_result_super, cut_result_super1, src_mask, dst_mask)
            """
            cv2.imwrite(f"result5/{i}_1.jpg", final_result_fast)
            """
            cv2.imwrite(f"result5/{i}_2.jpg", final_result_normal)
            cv2.imwrite(f"result5/{i}_3.jpg", final_result_super)
            cv2.imwrite(f"result5/{i}_4.jpg", final_result_apap)
            """

            """

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            final_result_fast_gray = cv2.cvtColor(final_result_fast, cv2.COLOR_BGR2GRAY)
            final_result_normal_gray = cv2.cvtColor(final_result_normal, cv2.COLOR_BGR2GRAY)
            final_result_apap_gray = cv2.cvtColor(final_result_apap, cv2.COLOR_BGR2GRAY)
            final_result_super_gray = cv2.cvtColor(final_result_super, cv2.COLOR_BGR2GRAY)

            _, mask1 = cv2.threshold(src=final_result_fast_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            fast_psnr = Psnr.compute_psnr(frame_copy, final_result_fast)
            fast_ssim, fast_l = Ssim.ssim1(frame_copy, final_result_fast)
            Our_PSNR.append(round(fast_psnr, 4))
            Our_SSIM.append(round(fast_ssim, 4))
            print("Our PSNR:", fast_psnr)
            print("Our SSIM:", fast_ssim)

            _, mask1 = cv2.threshold(src=final_result_normal_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_gray1 = frame_gray.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            normal_psnr = Psnr.compute_psnr(frame_copy, final_result_normal)
            normal_ssim, normal_l = Ssim.ssim1(frame_copy, final_result_normal)
            Normal_PSNR.append(round(normal_psnr, 4))
            Normal_SSIM.append(round(normal_ssim, 4))
            print("Normal PSNR:", normal_psnr)
            print("Normal SSIM:", normal_ssim)

            _, mask1 = cv2.threshold(src=final_result_super_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            super_psnr = Psnr.compute_psnr(frame_copy, final_result_super)
            super_ssim, super_l = Ssim.ssim1(frame_copy, final_result_super)
            Super_PSNR.append(round(super_psnr, 4))
            Super_SSIM.append(round(super_ssim, 4))
            print("Super PSNR:", super_psnr)
            print("Super SSIM:", super_ssim)

            _, mask1 = cv2.threshold(src=final_result_apap_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            apap_psnr = Psnr.compute_psnr(frame_copy, final_result_apap)
            apap_ssim, apap_l = Ssim.ssim1(frame_copy, final_result_apap)
            Apap_PSNR.append(round(apap_psnr, 4))
            Apap_SSIM.append(round(apap_ssim, 4))
            print("Apap PSNR:", apap_psnr)
            print("Apap SSIM:", apap_ssim)
            """
            final_result_fast = final_result_fast.astype(np.uint8)
            """
            final_result_normal = final_result_normal.astype(np.uint8)
            final_result_super = final_result_super.astype(np.uint8)
            final_result_apap = final_result_apap.astype(np.uint8)
            """
            video_writer.write(final_result_fast)
            """
            video_writer1.write(final_result_normal)
            video_writer2.write(final_result_super)
            video_writer3.write(final_result_apap)
            """
            print(f"第{i}帧完成")
            i += 1
        elif frame_count % frame_interval == 0 and i > 400:
            """
            (b, g, r) = cv2.split(frame)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
            bh = clahe.apply(b)
            gh = clahe.apply(g)
            rh = clahe.apply(r)
            frame = cv2.merge((bh, gh, rh), )
            """
            """
            result_fast, result_normal, result_apap = map2(frame, frame1, Hg1, mask1, mask, "right_bottom", i)
            result_fast1, result_normal1, result_apap1  = map2(frame, frame2, Hg2, mask2, mask, "right_top", i)
            result_fast2, result_normal2, result_apap2  = map2(frame, frame3, Hg3, mask3, mask, "left_bottom", i)
            result_fast3, result_normal3, result_apap3  = map2(frame, frame4, Hg4, mask4, mask, "left_top", i)
            result_fast2 = histogram.Histogram_Matching(result_fast2, result_fast3)
            result_fast1 = histogram.Histogram_Matching(result_fast1, result_fast3)
            result_fast = histogram.Histogram_Matching(result_fast, result_fast1)
            cut_result_fast, src_mask1, dst_mask1 = seamcut1(result_fast, result_fast1)
            cut_result_fast1, src_mask2, dst_mask2 = seamcut1(result_fast2, result_fast3)
            final_result_fast, src_mask, dst_mask = seamcut1(cut_result_fast, cut_result_fast1)

            """
            result_fast, result_normal, result_apap = map2(frame, frame1, Hg1, mask1, mask, "right_bottom", i)
            #fast_right_bottom_rmse.append(rmse)
            #normal_right_bottom_rmse.append(rmse1)
            #fast_right_bottom_mse.append(mse)
            #normal_right_bottom_mse.append(mse1)
            result_fast1, result_normal1, result_apap1 = map2(frame, frame2, Hg2, mask2, mask, "right_top", i)
            #fast_right_top_rmse.append(rmse)
            #normal_right_top_rmse.append(rmse1)
            #fast_right_top_mse.append(mse)
            #normal_right_top_mse.append(mse1)
            result_fast2, result_normal2, result_apap2 = map2(frame, frame3, Hg3, mask3, mask, "left_bottom", i)
            #fast_left_bottom_rmse.append(rmse)
            #normal_left_bottom_rmse.append(rmse1)
            #fast_left_bottom_mse.append(mse)
            #normal_left_bottom_mse.append(mse1)
            result_fast3, result_normal3, result_apap3 = map2(frame, frame4, Hg4, mask4, mask, "left_top", i)
            #fast_left_top_rmse.append(rmse)
            #normal_left_top_rmse.append(rmse1)
            #fast_left_top_mse.append(mse)
            #normal_left_top_mse.append(mse1)

            """
            result_super, hg, com_image, same, rmse, mse = input2(frame, frame1)
            #super_right_bottom_rmse.append(rmse)
            #super_right_bottom_mse.append(mse)
            result_super1, hg, com_image, same, rmse, mse = input2(frame, frame2)
            #super_right_top_rmse.append(rmse)
            #super_right_top_mse.append(mse)
            result_super2, hg, com_image, same, rmse, mse = input2(frame, frame3)
            #super_left_bottom_rmse.append(rmse)
            #super_left_bottom_mse.append(mse)
            result_super3, hg, com_image, same, rmse, mse = input2(frame, frame4)
            #super_left_top_rmse.append(rmse)
            #super_left_top_mse.append(mse)
            """
            """
            result_fast2 = histogram.Histogram_Matching(result_fast2, result_fast3)
            result_fast1 = histogram.Histogram_Matching(result_fast1, result_fast3)
            result_fast = histogram.Histogram_Matching(result_fast, result_fast1)

            result_normal2 = histogram.Histogram_Matching(result_normal2, result_normal3)
            result_normal1 = histogram.Histogram_Matching(result_normal1, result_normal3)
            result_normal = histogram.Histogram_Matching(result_normal, result_normal1)

            result_apap2 = histogram.Histogram_Matching(result_apap2, result_apap3)
            result_apap1 = histogram.Histogram_Matching(result_apap1, result_apap3)
            result_apap = histogram.Histogram_Matching(result_apap, result_apap1)

            result_super2 = histogram.Histogram_Matching(result_super2, result_super3)
            result_super1 = histogram.Histogram_Matching(result_super1, result_super3)
            result_super = histogram.Histogram_Matching(result_super, result_super1)
            """
            """
            result_fast1 = histogram.Histogram_Matching(result_fast1, result_fast)
            result_fast2 = histogram.Histogram_Matching(result_fast2, result_fast)
            result_fast3 = histogram.Histogram_Matching(result_fast3, result_fast2)
            """
        
            #数据集1的直方图方法
            result_fast = histogram.Histogram_Matching(result_fast, frame)
            result_fast1 = histogram.Histogram_Matching(result_fast1, frame)
            result_fast2 = histogram.Histogram_Matching(result_fast2, frame)
            result_fast3 = histogram.Histogram_Matching(result_fast3, frame)
            """
            result_normal = histogram.Histogram_Matching(result_normal, frame)
            result_normal1 = histogram.Histogram_Matching(result_normal1, frame)
            result_normal2 = histogram.Histogram_Matching(result_normal2, frame)
            result_normal3 = histogram.Histogram_Matching(result_normal3, frame)

            result_apap = histogram.Histogram_Matching(result_apap, frame)
            result_apap1 = histogram.Histogram_Matching(result_apap1, frame)
            result_apap2 = histogram.Histogram_Matching(result_apap2, frame)
            result_apap3 = histogram.Histogram_Matching(result_apap3, frame)

            result_super = histogram.Histogram_Matching(result_super, frame)
            result_super1 = histogram.Histogram_Matching(result_super1, frame)
            result_super2 = histogram.Histogram_Matching(result_super2, frame)
            result_super3 = histogram.Histogram_Matching(result_super3, frame)
            """
            cut_result_fast = seamcut2(result_fast, result_fast1, src_mask1, dst_mask1)
            cut_result_fast1 = seamcut2(result_fast2, result_fast3, src_mask2, dst_mask2)
            final_result_fast = seamcut2(cut_result_fast, cut_result_fast1, src_mask, dst_mask)
            """
            cut_result_normal = seamcut2(result_normal, result_normal1, src_mask1, dst_mask1)
            cut_result_normal1 = seamcut2(result_normal2, result_normal3, src_mask2, dst_mask2)
            final_result_normal = seamcut2(cut_result_normal, cut_result_normal1, src_mask, dst_mask)

            cut_result_apap = seamcut2(result_apap, result_apap1, src_mask1, dst_mask1)
            cut_result_apap1 = seamcut2(result_apap2, result_apap3, src_mask2, dst_mask2)
            final_result_apap = seamcut2(cut_result_apap, cut_result_apap1, src_mask, dst_mask)

            cut_result_super = seamcut2(result_super, result_super1, src_mask1, dst_mask1)
            cut_result_super1 = seamcut2(result_super2, result_super3, src_mask2, dst_mask2)
            final_result_super = seamcut2(cut_result_super, cut_result_super1, src_mask, dst_mask)
            """
            cv2.imwrite(f"result5/{i}_1.jpg", final_result_fast)
            """
            cv2.imwrite(f"result5/{i}_2.jpg", final_result_normal)
            cv2.imwrite(f"result5/{i}_3.jpg", final_result_super)
            cv2.imwrite(f"result5/{i}_4.jpg", final_result_apap)
            """
            """
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            final_result_fast_gray = cv2.cvtColor(final_result_fast, cv2.COLOR_BGR2GRAY)
            final_result_normal_gray = cv2.cvtColor(final_result_normal, cv2.COLOR_BGR2GRAY)
            final_result_apap_gray = cv2.cvtColor(final_result_apap, cv2.COLOR_BGR2GRAY)
            final_result_super_gray = cv2.cvtColor(final_result_super, cv2.COLOR_BGR2GRAY)

            _, mask1 = cv2.threshold(src=final_result_fast_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            fast_psnr = Psnr.compute_psnr(frame_copy, final_result_fast)
            fast_ssim, fast_l = Ssim.ssim1(frame_copy, final_result_fast)
            Our_PSNR.append(round(fast_psnr, 4))
            Our_SSIM.append(round(fast_ssim, 4))
            print("Our PSNR:", fast_psnr)
            print("Our SSIM:", fast_ssim)

            _, mask1 = cv2.threshold(src=final_result_normal_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            normal_psnr = Psnr.compute_psnr(frame_copy, final_result_normal)
            normal_ssim, normal_l = Ssim.ssim1(frame_copy, final_result_normal)
            Normal_PSNR.append(round(normal_psnr, 4))
            Normal_SSIM.append(round(normal_ssim, 4))
            print("Normal PSNR:", normal_psnr)
            print("Normal SSIM:", normal_ssim)

            _, mask1 = cv2.threshold(src=final_result_super_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            super_psnr = Psnr.compute_psnr(frame_copy, final_result_super)
            super_ssim, super_l = Ssim.ssim1(frame_copy, final_result_super)
            Super_PSNR.append(round(super_psnr, 4))
            Super_SSIM.append(round(super_ssim, 4))
            print("Super PSNR:", super_psnr)
            print("Super SSIM:", super_ssim)

            _, mask1 = cv2.threshold(src=final_result_apap_gray, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
            frame_copy = frame.copy()
            frame_copy[mask1==0] = [0, 0, 0]
            apap_psnr = Psnr.compute_psnr(frame_copy, final_result_apap)
            apap_ssim, apap_l = Ssim.ssim1(frame_copy, final_result_apap)
            Apap_PSNR.append(round(apap_psnr, 4))
            Apap_SSIM.append(round(apap_ssim, 4))
            print("Apap PSNR:", apap_psnr)
            print("Apap SSIM:", apap_ssim)
            """
            cv2.imwrite(f"result5/{i}.jpg", final_result_fast)
            final_result_fast = final_result_fast.astype(np.uint8)
            """
            final_result_normal = final_result_normal.astype(np.uint8)
            final_result_super = final_result_super.astype(np.uint8)
            final_result_apap = final_result_apap.astype(np.uint8)
            """
            video_writer.write(final_result_fast)
            """
            video_writer1.write(final_result_normal)
            video_writer2.write(final_result_super)
            video_writer3.write(final_result_apap)
            """
            print(f"第{i}帧完成")
            i += 1
        elif frame_count % frame_interval == 0 and i < 400:
            i += 1
        frame_count += 1
    """
    data = pd.DataFrame({
    "fast_right_bottom_rmse": fast_right_bottom_rmse,
    "normal_right_bottom_rmse": normal_right_bottom_rmse,
    "super_right_bottom_rmse": super_right_bottom_rmse,
    "fast_right_bottom_mse": fast_right_bottom_mse,
    "normal_right_bottom_mse": normal_right_bottom_mse,
    "super_right_bottom_mse": super_right_bottom_mse,
    "fast_right_top_rmse": fast_right_top_rmse,
    "normal_right_top_rmse": normal_right_top_rmse,
    "super_right_top_rmse": super_right_top_rmse,
    "fast_right_top_mse": fast_right_top_mse,
    "normal_right_top_mse": normal_right_top_mse,
    "super_right_top_mse": super_right_top_mse,
    "fast_left_bottom_rmse": fast_left_bottom_rmse,
    "normal_left_bottom_rmse": normal_left_bottom_rmse,
    "super_left_bottom_rmse": super_left_bottom_rmse,
    "fast_left_bottom_mse": fast_left_bottom_mse,
    "normal_left_bottom_mse": normal_left_bottom_mse,
    "super_left_bottom_mse": super_left_bottom_mse,
    "fast_left_top_rmse": fast_left_top_rmse,
    "normal_left_top_rmse": normal_left_top_rmse,
    "super_left_top_rmse": super_left_top_rmse,
    "fast_left_top_mse": fast_left_top_mse,
    "normal_left_top_mse": normal_left_top_mse,
    "super_left_top_mse": super_left_top_mse,
    })
    data.to_excel("result5/output.xlsx", index=False)
    """
    video.release()
    video1.release()
    video2.release()
    video3.release()
    video4.release()
