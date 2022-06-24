#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
