#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def dim3_distance(vec1, vec2):
    return sum((vec1 - vec2) ** 2)


def camera_correspondence(cam_list):
    vec_list = []
    for i, cam in enumerate(cam_list):
        cam_list[i].para_load()
        vec_list.append(cam_list[i].cam_world_cood)

    pair_list = []
    for i, vec1 in enumerate(vec_list):
        for j, vec2 in enumerate(vec_list):
            if i == j or i > j:
                continue
            elif dim3_distance(vec1, vec2) < 2:
                pair_list.append((i, j))

    return pair_list


def SS_mat(vec3):
    vec3 = np.squeeze(vec3)
    SS_mat = np.zeros((3, 3))
    SS_mat[0, 1] = -vec3[2]
    SS_mat[0, 2] = vec3[1]
    SS_mat[1, 0] = vec3[2]
    SS_mat[1, 2] = -vec3[0]
    SS_mat[2, 0] = -vec3[1]
    SS_mat[2, 1] = vec3[0]
    return SS_mat


def FF_mat(A1, A2, Rt1, Rt2):
    P1 = np.dot(A1, Rt1[0:3, 0:4])
    P2 = np.dot(A2, Rt2[0:3, 0:4])
    cam_pos1 = -np.dot(Rt1[0:3, 0:3].T, Rt1[0:3, 3])
    cam_pos1 = np.array([cam_pos1[0], cam_pos1[1], cam_pos1[2], 1])
    epipole2 = np.dot(P2, cam_pos1)
    cam_pos2 = -np.dot(Rt2[0:3, 0:3].T, Rt2[0:3, 3])
    cam_pos2 = np.array([cam_pos2[0], cam_pos2[1], cam_pos2[2], 1])
    epipole1 = np.dot(P1, cam_pos2)
    return epipole1, epipole2, np.dot(SS_mat(epipole2), np.dot(P2, np.linalg.pinv(P1)))


def gene(angles):
    # 正規化
    B = list(map(lambda y: y - min(angles), angles))
    return list(map(lambda y: (y - min(B)) / (max(B) - min(B)), B))


def epipole_angle(img_num, epipole_dict, cam_list=[]):
    """エピポールと輪郭上の点を結んだ直線と，x軸のなす角
    Parameters
    ========================
    img_num: int
    epipole_dict: dict
    cam_list: list

    Returns
    ========================
    angle_list: list

    """
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


def epilines_para(frags, F):

    lines_list = []
    for color in frags:
        temp_color_list = []
        for frag in color:
            frag_lines = cv2.computeCorrespondEpilines(
                frag.reshape(-1, 1, 2), 1, F
            )  # ndarray(フラグメントの座標数,1,3)
            temp_color_list.append(frag_lines)

        lines_list.append(temp_color_list)

    return lines_list


def get_frag_cood(frag_list):
    cood_S = []
    cood_F = []
    for color in frag_list:
        col_S = np.array([frag[0] for frag in color])
        col_F = np.array([frag[-1] for frag in color])
        cood_S.append(col_S)
        cood_F.append(col_F)
    return cood_S, cood_F  # cood_S[color][frag]


def para2cood_S(para_list):
    return np.array([[0, -c / b] for a, b, c in para_list])


def para2cood_F(para_list):
    """
    Todo: ハードコードされている数値のパラメータへの移行
    """
    return np.array([[1920, -(1920 * a + c) / b] for a, b, c in para_list])


def all_pa2co(para_list):
    epi_cood_S = []
    epi_cood_F = []
    for color in para_list:
        color_list_S = []
        color_list_F = []
        for frag in color:
            S_cood = para2cood_S(frag.squeeze())
            F_cood = para2cood_F(frag.squeeze())
            color_list_S.append(S_cood)
            color_list_F.append(F_cood)
        epi_cood_S.append(color_list_S)
        epi_cood_F.append(color_list_F)
    return epi_cood_S, epi_cood_F  # epi_cood[color][frag]


def coll_t1_t2(epi_cood_S, epi_cood_F, cood_S, cood_F):
    epi_cood_S_bro = np.repeat(epi_cood_S, len(cood_S), axis=0).reshape(
        (epi_cood_S.shape[0], len(cood_S), epi_cood_S.shape[1])
    )
    epi_cood_F_bro = np.repeat(epi_cood_F, len(cood_S), axis=0).reshape(
        (epi_cood_F.shape[0], len(cood_S), epi_cood_F.shape[1])
    )
    v = cood_S - epi_cood_S_bro
    v2 = cood_F - cood_S
    v1 = epi_cood_F_bro - epi_cood_S_bro
    t1 = np.cross(v, v2) / np.cross(v1, v2)
    t2 = np.cross(v, v1) / np.cross(v1, v2)
    return t1, t2


def coll_det(t1, t2):
    t1_t = np.array((t1 <= 1) & (t1 > 0), dtype=np.int16)
    t2_t = np.array((t2 <= 1) & (t2 > 0), dtype=np.int16)
    count_c = np.array(t1_t + t2_t == 2, dtype=np.int64)
    # surport_idx = np.argmax(np.sum(count_c,axis=0))
    count_c = np.sum(count_c, axis=0)
    sorted_count_c = np.argsort(count_c)
    count_c = np.where(sorted_count_c > np.max(sorted_count_c) - 10)[0]
    return count_c  # surport_idx


def make_piar_list(epi_cood_S, epi_cood_F, cood_S, cood_F):
    """
    Todo: typo? piar->pair
    """
    img_list = []
    for epi_S_col, epi_F_col, S_col, F_col in zip(
        epi_cood_S, epi_cood_F, cood_S, cood_F
    ):
        color_list = []
        if len(epi_S_col) == 0 or len(S_col) == 0:
            img_list.append(color_list)
            continue
        for epi_S_frag, epi_F_frag in zip(epi_S_col, epi_F_col):
            t1, t2 = coll_t1_t2(epi_S_frag, epi_F_frag, S_col, F_col)
            surport_idx = coll_det(t1, t2)
            color_list.append(surport_idx)
        img_list.append(color_list)
    return img_list


def pair_and_key_gen(
    pair,
    cam_list=[],
    cam_pairs_F=[],
):
    pair_list = {}
    F = cam_pairs_F[pair]
    frags_para12 = epilines_para(cam_list[pair[0]].frag_list, F)  # frags_para[色][frag]
    frags_para21 = epilines_para(cam_list[pair[1]].frag_list, F.T)

    cood_S, cood_F = get_frag_cood(cam_list[pair[1]].frag_list)
    epi_cood_S, epi_cood_F = all_pa2co(frags_para12)
    img_list1 = make_piar_list(epi_cood_S, epi_cood_F, cood_S, cood_F)

    cood_S, cood_F = get_frag_cood(cam_list[pair[0]].frag_list)
    epi_cood_S, epi_cood_F = all_pa2co(frags_para21)
    img_list2 = make_piar_list(epi_cood_S, epi_cood_F, cood_S, cood_F)

    pair_list[((pair[0], pair[1]), "F")] = img_list1
    pair_list[((pair[0], pair[1]), "R")] = img_list2
    return pair_list
