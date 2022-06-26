#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from base import normalization


def nom_F(F):
    """
    Todo: sum -> np.sumの方が良い？
    """
    return (1 / sum(sum(F**2)) ** (1 / 2)) * F


def cover_mat(x1, y1, x2, y2):
    return np.array(
        [
            [x1**2 + x2**2, x2 * y2, x2, x1 * y1, 0, 0, x1, 0, 0],
            [x2 * y2, x1**2 + y2**2, y2, 0, x1 * y1, 0, 0, x1, 0],
            [x2, y2, 1, 0, 0, 0, 0, 0, 0],
            [x1 * y1, 0, 0, y1**2 + x2**2, x2 * y2, x2, y1, 0, 0],
            [0, x1 * y1, 0, x2 * y2, y1**2 + y2**2, y2, 0, y1, 0],
            [0, 0, 0, x2, y2, 1, 0, 0, 0],
            [x1, 0, 0, y1, 0, 0, 1, 0, 0],
            [0, x1, 0, 0, y1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


def min_dist(F, pt1, pt2):
    """
    Parameters
    ===================
    pt1が画像2上の点，pt2が画像1上の点

    Todo: typo? -> thita
    """
    S0 = 10**10
    x1_ori = pt1[0]
    y1_ori = pt1[1]
    x2_ori = pt2[0]
    y2_ori = pt2[1]

    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]

    x1_tilda = 0
    y1_tilda = 0
    x2_tilda = 0
    y2_tilda = 0
    thita = nom_F(F).flatten()
    it = 0
    while True:
        V_eps = cover_mat(x1, y1, x2, y2)
        eps_ast = np.array(
            [
                x1 * x2 + x2 * x1_tilda + x2 * x2_tilda,
                x1 * y2 + y2 * x1_tilda + x2 * y2_tilda,
                x1 + x1_tilda,
                y1 * x2 + x2 * y1_tilda + y1 * x2_tilda,
                y1 * y2 + y2 * y1_tilda + y1 * y2_tilda,
                y1 + y1_tilda,
                x2 + x2_tilda,
                y2 + y2_tilda,
                1,
            ]
        )

        x1_y1_tilda = (
            np.dot(eps_ast, thita)
            * np.dot(
                np.array(
                    [[thita[0], thita[1], thita[2]], [thita[3], thita[4], thita[5]]]
                ),
                np.array([x2, y2, 1]),
            )
            / np.dot(thita, np.dot(V_eps, thita))
        )
        x2_y2_tilda = (
            np.dot(eps_ast, thita)
            * np.dot(
                np.array(
                    [[thita[0], thita[3], thita[6]], [thita[1], thita[4], thita[7]]]
                ),
                np.array([x1, y1, 1]),
            )
            / np.dot(thita, np.dot(V_eps, thita))
        )

        x1_tilda = x1_y1_tilda[0]
        y1_tilda = x1_y1_tilda[1]
        x2_tilda = x2_y2_tilda[0]
        y2_tilda = x2_y2_tilda[1]

        x1 = x1_ori - x1_tilda
        y1 = y1_ori - y1_tilda
        x2 = x2_ori - x2_tilda
        y2 = y2_ori - y2_tilda

        S = x1_tilda**2 + y1_tilda**2 + x2_tilda**2 + y2_tilda**2

        if abs(S0 - S) < 0.00001:
            break

        elif it == 20:
            break

        else:
            S0 = S
            it += 1

    return np.array((x1, y1)), np.array((x2, y2))


def Ps(P, pt):
    a = P[0, 0] - pt[0] * P[2, 0]
    b = P[0, 1] - pt[0] * P[2, 1]
    c = P[0, 2] - pt[0] * P[2, 2]
    d = P[0, 3]
    e = pt[0] * P[2, 3]
    f = P[1, 0] - pt[1] * P[2, 0]
    g = P[1, 1] - pt[1] * P[2, 1]
    h = P[1, 2] - pt[1] * P[2, 2]
    i = P[1, 3]
    j = pt[1] * P[2, 3]
    return a, b, c, d, e, f, g, h, i, j


def tri(P1, P2, pt1, pt2):
    # x = sympy.Symbol('x')
    # y = sympy.Symbol('y')
    # z = sympy.Symbol('z')
    a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = Ps(P1, pt1)
    a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = Ps(P2, pt2)
    T = np.array([[a1, b1, c1], [f1, g1, h1], [a2, b2, c2], [f2, g2, h2]])
    p = np.array([[d1 - e1], [i1 - j1], [d2 - e2], [i2 - j2]])
    T_inv = np.linalg.pinv(T)
    result_pt = np.dot(T_inv, -p)
    return result_pt


def excluded_Parray(ex_tag, cam_list=[]):
    P_dict = {}
    for i, cam in enumerate(cam_list):
        if i in ex_tag:
            continue
        P_dict[i] = cam.P
    return P_dict


def dot_P_frag(P, frag):
    repro_frag = []
    for pt in frag:
        repro_pt = np.dot(P, pt)
        repro_pt = np.array(normalization(repro_pt))
        repro_frag.append(repro_pt)
    return np.array(repro_frag)


def connect_contour(contour_list):
    con_list = []
    for col in contour_list:
        if len(col) == 0:
            con_list.append(np.empty((1, 2)))
            continue
        A = np.concatenate(col).reshape((-1, 2))
        con_list.append(A)
    return con_list


def cal_distance(repro_P, contour_P):
    contour_P = connect_contour(contour_P)
    distance_list = []
    for repro_col, con_col in zip(repro_P, contour_P):
        col_list = []
        for repro_frag in repro_col:
            repro_frag_bro = np.repeat(repro_frag, len(con_col), axis=0).reshape(
                (repro_frag.shape[0], len(con_col), repro_frag.shape[1])
            )
            distance = (np.sum((con_col - repro_frag_bro) ** 2, axis=2)) ** (1 / 2)
            col_list.append(distance)
        distance_list.append(col_list)
    return distance_list


def distance_check(distance_list):
    dist_check_list = []
    ac_list = []
    for col in distance_list:
        col_list = []
        ac_col_list = []
        for frag in col:
            ac = np.array((np.min(frag, axis=1)) < 5, dtype=np.int64)  # 条件:10 pixel以内
            col_list.append(sum(ac) / len(ac))
            ac_col_list.append(ac)
        ac_list.append(ac_col_list)
        dist_check_list.append(np.array(col_list))
    return ac_list, dist_check_list


def P_dict_check(repro_dict_taged, cam_list=[]):
    P_list = []
    P_ac_list = []
    for P_tag in repro_dict_taged:
        repro_P = repro_dict_taged[P_tag]
        contour_P = cam_list[P_tag].contour_list
        distance_list = cal_distance(repro_P, contour_P)
        ac_list, dist_check_list = distance_check(distance_list)
        P_list.append(dist_check_list)
        P_ac_list.append(ac_list)
    P_check = np.array(P_list)
    return ac_list, P_ac_list, P_check


def P_check_integration(P_check):
    check_list = []
    for col in range(P_check.shape[1]):
        temp_list = []
        for img in P_check[:, col]:
            temp = np.array(img > 0.8, dtype=np.int64)  # 曲線中の何割が閾値以内か
            temp_list.append(temp)
        col_check = np.sum(np.array(temp_list), axis=0)
        check_list.append(col_check)
    return check_list


def ac_list_integration(P_ac_list):
    inter_ac = []
    for i, col in enumerate(P_ac_list[0]):
        col_list = []
        for j in range(len(col)):
            temp_array = np.zeros(len(P_ac_list[0][i][j]))
            for img in P_ac_list:
                temp_array += img[i][j]
            col_list.append(temp_array)
        inter_ac.append(col_list)
    return inter_ac


def gen_support_dict(reprojection_dict, cam_list=[]):
    support_dict = {}
    for tag in reprojection_dict:
        repro_dict_taged = reprojection_dict[tag]
        _, P_ac_list, P_check = P_dict_check(repro_dict_taged, cam_list=cam_list)
        check_list = P_check_integration(P_check)
        inter_ac = ac_list_integration(P_ac_list)
        support_dict[tag] = (check_list, inter_ac)
    return support_dict
