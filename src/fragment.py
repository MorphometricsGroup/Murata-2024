#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def split_list(contour_length, max_frag_len=100, min_frag_len=40, min_overrap=20):

    # 輪郭のフラグメントの位置を指定(最小40 pixl)
    if contour_length > max_frag_len:
        pass

    elif contour_length < min_frag_len:
        return None

    elif contour_length == min_frag_len:
        return [[0, min_frag_len - 1]]

    else:
        max_frag_len = contour_length

    step0 = np.random.randint(min_frag_len, max_frag_len)  # 一つ目のフラグメントの長さ（40から100）
    sep_list = [[0, step0]]
    back = np.random.randint(min_overrap, step0 - 1)  # フラグメントを重ねるために戻す分を決める（最小10 pixl）
    next_start = step0 - back

    while True:

        # 戻った分(back)より進む
        if back + 1 > min_frag_len:
            step = np.random.randint(back + 1, max_frag_len)
        else:
            step = np.random.randint(min_frag_len, max_frag_len)

        full_length = next_start + step
        sept = [next_start, full_length]
        sep_list.append(sept)
        back = np.random.randint(min_overrap, step - 1)
        next_start = full_length - back

        # 終了判定
        if full_length > contour_length:
            break

    # 超過した分戻す（長さはそのまま）
    difference = sep_list[-1][1] - (contour_length - 1)
    sep_list[-1][0] -= difference
    sep_list[-1][1] -= difference

    return sep_list


def contours_split(contour):

    # contour.shape == (N, 2)
    contour_length = contour.shape[0]
    sp_list = split_list(contour_length)

    if sp_list == None:
        return None

    frag_list = []
    # 位置のリスト通りにスライス
    for sp in sp_list:
        # print(sp)
        frag_list.append(contour[sp[0] : sp[1], :])

    return frag_list


def all_fraged(contours_list):

    # 輪郭のリストからフラグメントのリストを得る
    frags_list = []

    # for i in contours_list:
    # temp_list = []
    frags = []
    for j in contours_list:
        temp_frags = contours_split(j.squeeze())

        if temp_frags != None:
            frags += temp_frags

    # if frags != []:
    #    frags_list.append(frags)

    return frags


def frag_list_fraged(frags_list):  # frags_list[色][輪郭][sep][座標]
    img_frag_list = []
    for frag in frags_list:
        color_frag = all_fraged(frag)
        img_frag_list.append(color_frag)
    return img_frag_list


# エピポールと輪郭上の点を結んだ直線と，x軸のなす角
def gene(angles):
    # 正規化
    B = list(map(lambda y: y - min(angles), angles))
    return list(map(lambda y: (y - min(B)) / (max(B) - min(B)), B))


def epipole_angle(img_num, epipole_dict):
    cam = cam_list[img_num]
    cam.img_load()
    cam.contour_extraction()
    angle_list = []

    for epi in epipole_dict[img_num]:
        epi_angle_list = []
        epi_x = epi[0]
        epi_y = epi[1]
        for color in cam.contour_list:
            color_angle_list = []
            for contour in color:
                contour_angle_list = []
                for cood in contour:
                    x = cood[0][0]
                    y = cood[0][1]
                    tilt = (y - epi_y) / (x - epi_x)
                    angle = np.arctan(tilt)
                    contour_angle_list.append(angle)
                contour_angle_list = gene(contour_angle_list)
                color_angle_list.append(contour_angle_list)
            epi_angle_list.append(color_angle_list)
        angle_list.append(epi_angle_list)
    return angle_list


def expand(idx_l, list_length):
    del_list = []
    for i in idx_l:
        if np.isnan(i):
            continue
        if i - 2 < 0:
            del_list.append(list_length + i - 2)
        else:
            del_list.append(i - 2)

        if i - 1 < 0:
            del_list.append(list_length + i - 1)
        else:
            del_list.append(i - 1)

        del_list.append(i)

        if i + 1 > list_length - 1:
            del_list.append(i + 1 - list_length)
        else:
            del_list.append(i + 1)
        if i + 2 > list_length - 1:
            del_list.append(i + 2 - list_length)
        else:
            del_list.append(i + 2)
    return sorted(list(set(del_list)))


def differential(angles):
    # エピポーラ線に平行な接線をもつ点(前後方微分の正負を比べたほうが良い)
    del_idx = []
    for i in range(len(angles)):
        if np.isnan(angles[i]):
            continue
        if i == len(angles) - 1:
            if np.sign(angles[i] - angles[i - 1]) != np.sign(
                angles[0] - angles[i]
            ):  # or abs(angles[0]-angles[i-1])/2 < 0.001:
                del_idx.append(i)
        else:
            if np.sign(angles[i] - angles[i - 1]) != np.sign(
                angles[i + 1] - angles[i]
            ):  # or abs(angles[i+1]-angles[i-1])/2 < 0.001:
                del_idx.append(i)
    # del_idx = expand(del_idx, len(angles))

    return del_idx


def marge_del(epi_del_list):
    im_del_list = []
    for i, col in enumerate(epi_del_list[0]):
        color_list = []
        for j, con in enumerate(col):
            color_list.append(list(set(epi_del_list[0][i][j] + epi_del_list[1][i][j])))
        im_del_list.append(color_list)
    return im_del_list


def all_D(angles_list):
    # 画像1枚に対して削除リストを作成
    all_del_list = []
    for epi in angles_list:
        epi_del_list = []
        for color in epi:
            color_del_list = []
            for contour in color:
                # if len(contour)<40:
                #    continue
                del_idx = differential(contour)
                color_del_list.append(del_idx)
            epi_del_list.append(color_del_list)
        all_del_list.append(epi_del_list)
    all_del_list = marge_del(all_del_list)
    return all_del_list
