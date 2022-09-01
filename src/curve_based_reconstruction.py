#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix, coo_matrix


def FR_frags(dict_tag, cam_list=[]):
    if dict_tag[1] == "F":
        part = cam_list[dict_tag[0][0]].frag_list
        counterpart = cam_list[dict_tag[0][1]].frag_list
        return part, counterpart

    elif dict_tag[1] == "R":
        part = cam_list[dict_tag[0][1]].frag_list
        counterpart = cam_list[dict_tag[0][0]].frag_list
        return part, counterpart


# 座標でdictを作る
def coordinate_dict_gen(tag, cam_list=[], dest_dir="temp"):
    """_summary_

    Parameters
    ----------
    tag : _type_
        _description_
    cam_list : list, optional
        _description_, by default []
    dest_dir : str, optional
        _description_, by default "temp"

    X_Y_Z.coordinate_dict: list of list, of shape (n_labels, n_pairs), of ndarray, of shape (n_cams, n_intersections, n_dim)
        X: cam1のid
        Y: cam2のid
        Z: F or R
        n_labels: ラベルの数
        n_pairs: ペアの数（画像間で最大10個まで）
        n_cams: cam1とcam2
        n_intersections: cam1からエピポーラ線とcam2のfragmentの交点の数
        n_dim: 画像の次元, 画素のi,j座標

        TODO: 画像id, fragment id, 画素idで座標値を指定できるリストかndarrayとして扱う

    """

    pair_coordinate = []
    part, counterpart = FR_frags(tag, cam_list)

    pair_list_file_path = os.path.join(
        dest_dir, r"{0}_{1}_{2}.pair_list".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(pair_list_file_path, "rb") as f:
        pair_list_taged = pickle.load(f)

    pair_pt_file_path = os.path.join(
        dest_dir, r"{0}_{1}_{2}.pair_pt".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(pair_pt_file_path, "rb") as f:
        pair_pt_taged = pickle.load(f)

    for part_col, cpart_col, pair_col, PtPair_col in zip(
        part, counterpart, pair_list_taged, pair_pt_taged
    ):
        col_list = []
        for part_frag, pair, pt_idx in zip(part_col, pair_col, PtPair_col):
            for each_pair, each_pt_idx in zip(pair, pt_idx):
                if each_pt_idx[0].size != 0:
                    col_list.append(
                        (
                            np.array(
                                [
                                    part_frag[each_pt_idx[0]],
                                    cpart_col[each_pair][each_pt_idx[1]],
                                ]
                            )
                        )
                    )
        pair_coordinate.append(col_list)

    os.remove(pair_list_file_path)
    os.remove(pair_pt_file_path)

    coordinate_dict_file_path = os.path.join(
        dest_dir, r"{0}_{1}_{2}.coordinate_dict".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(coordinate_dict_file_path, "wb") as f:
        pickle.dump(pair_coordinate, f)


def FR_check(dict_tag, cam_list=[], cam_pairs_F=[]):
    if dict_tag[1] == "F":
        P1 = cam_list[dict_tag[0][0]].P
        P2 = cam_list[dict_tag[0][1]].P
        F = cam_pairs_F[dict_tag[0]]
        return P1, P2, F
    elif dict_tag[1] == "R":
        P1 = cam_list[dict_tag[0][1]].P
        P2 = cam_list[dict_tag[0][0]].P
        F = cam_pairs_F[dict_tag[0]].T
        return P1, P2, F


def nom_F(F):
    """
    Todo: sum -> np.sumの方が良い？
    """
    return (1 / np.sum(np.sum(F**2)) ** (1 / 2)) * F


def cover_mat(x1, y1, x2, y2):
    """共分散行列
    Todo: scipyでの置き換え
    """
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
    """観測した対応点を，視線が交わるように最短に補正する
    Parameters
    ===================
    pt1が画像2上の点，pt2が画像1上の点
    F: fundamental matrix

    Returns
    ========================
    np.array((x1, y1)): 補正された画像2の点
    np.array((x2, y2)): 補正された画像1の点
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
    theta = nom_F(F).flatten()
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
            np.dot(eps_ast, theta)
            * np.dot(
                np.array(
                    [[theta[0], theta[1], theta[2]], [theta[3], theta[4], theta[5]]]
                ),
                np.array([x2, y2, 1]),
            )
            / np.dot(theta, np.dot(V_eps, theta))
        )
        x2_y2_tilda = (
            np.dot(eps_ast, theta)
            * np.dot(
                np.array(
                    [[theta[0], theta[3], theta[6]], [theta[1], theta[4], theta[7]]]
                ),
                np.array([x1, y1, 1]),
            )
            / np.dot(theta, np.dot(V_eps, theta))
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
    """三角測量
    Parameters
    ===================
    P: perspective matrix
    F: fundamental matrix
    pt: correspondence points

    Returns
    ========================
    result_pt: 3 vector
    """
    a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = Ps(P1, pt1)
    a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = Ps(P2, pt2)
    T = np.array([[a1, b1, c1], [f1, g1, h1], [a2, b2, c2], [f2, g2, h2]])
    p = np.array([[d1 - e1], [i1 - j1], [d2 - e2], [i2 - j2]])
    T_inv = np.linalg.pinv(T)
    result_pt = np.dot(T_inv, -p)
    return result_pt


def TDlines_gen(tag, cam_list=[], cam_pairs_F=[], src_dir="temp"):

    src_file_path = os.path.join(
        src_dir, r"{0}_{1}_{2}.coordinate_dict".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(src_file_path, "rb") as f:
        pts = pickle.load(f)

    P1_ori, P2_ori, F_ori = FR_check(tag, cam_list=cam_list, cam_pairs_F=cam_pairs_F)
    # pt, sep_list = connect_points(pts)
    temp_TDlines = []
    for pts_col in pts:
        col_list = []
        for pt in pts_col:
            pt = np.transpose(pt, (1, 0, 2))
            F = np.broadcast_to(F_ori, (pt.shape[0], 3, 3))
            P1 = np.broadcast_to(P1_ori, (pt.shape[0], 3, 4))
            P2 = np.broadcast_to(P2_ori, (pt.shape[0], 3, 4))
            newcoords = np.array(list(map(min_dist, F, pt[:, 1, :], pt[:, 0, :])))
            tri_pt = np.array(
                list(map(tri, P1, P2, newcoords[:, 1, :], newcoords[:, 0, :]))
            )
            # pts_array = sep_array(tri_pt, sep_list)
            col_list.append(tri_pt)
        temp_TDlines.append(col_list)

    os.remove(r"temp/{0}_{1}_{2}.coordinate_dict".format(tag[0][0], tag[0][1], tag[1]))
    with open(
        r"temp/{0}_{1}_{2}.TDlines".format(tag[0][0], tag[0][1], tag[1]), "wb"
    ) as f:
        pickle.dump(temp_TDlines, f)
    # TDlines[j] = temp_TDlines


def excluded_Parray(ex_tag, cam_list=[]):
    """再投影時に必要のないPを除外したdictを作る
    Parameters
    ===================
    ex_tag: tuple of shape (2)
        tuple of camera_pair

    Returns
    ========================
    P_dict: dict
        key: camera_num
        val: P
    """
    P_dict = {}
    for i, cam in enumerate(cam_list):
        if i in ex_tag:
            continue
        P_dict[i] = cam.P
    return P_dict


def dot_P_frag(P, frag):
    """再投影の計算
    Parameters
    ===================
    P: perspective matrix
    frag: ndarray of shape (n_intersections, 3)
        3D fragmentの配列（交点座標を持つ）

    Returns
    ========================
    np.array(repro_frag): 2D fragment
    """
    frag_ = np.concatenate(
        [frag, np.ones(len(frag)).reshape((len(frag), 1))], 1
    )  # 末尾に1を追加 (X, Y, Z, 1)
    repro_frag = np.dot(P, frag_.transpose()).transpose()
    repro_frag = repro_frag[:, :2] / repro_frag[:, [-1]]
    return repro_frag.astype(np.int16)


def reprojection_gen(tag, cam_list=[], tmp_dir="temp"):
    """
    Parameters
    ===================
    tag: tuple of shape (2)
        val1: tuple of shape (2)
            tuple of camera_pair
        val2: str
            F or R

    X_Y_Z.TDlines: list of list, of shape (n_labels, n_pairs), of ndarray, of shape (n_intersections, n_dim, 1)
        X: cam1のid
        Y: cam2のid
        Z: F or R
        n_labels: ラベルの数
        n_pairs: ペアの数（画像間で最大10個まで）
        n_intersections: cam1からエピポーラ線とcam2のfragmentの交点の数
        n_dim: 3
            3次元空間中の座標のx,y,z座標

    X_Y_Z.reprojection_dict: dict
        X: cam1のid
        Y: cam2のid
        Z: F or R
        key: 再投影したカメラid
        val: list of list of list of shape (n_labels, n_fragments_3d) of ndarray, of shape(n_intersections, n_dim)
            n_labels: ラベルの数
            n_fragments_3d: 3次元空間中でのfragment（ペアの数に等しい）

            投影先の画像内での座標値



    """
    reprojection_dict = {}
    temp_reprojection_dict = {}
    P_dict = excluded_Parray(tag[0], cam_list=cam_list)
    for P_tag in P_dict:
        P = P_dict[P_tag]
        P_list = []
        TDline_file_path = os.path.join(
            tmp_dir, r"{0}_{1}_{2}.TDlines".format(tag[0][0], tag[0][1], tag[1])
        )
        with open(TDline_file_path, "rb") as f:
            TDlines_taged = pickle.load(f)
        for col in TDlines_taged:
            col_list = []
            for i, frag in enumerate(col):
                frag = frag.reshape((-1, 3))
                reprojection = dot_P_frag(P, frag)
                col_list.append(reprojection)
            P_list.append(col_list)
        temp_reprojection_dict[P_tag] = P_list
    # reprojection_dict[tag] = temp_reprojection_dict
    reprojection_dict_file_path = os.path.join(
        tmp_dir, r"{0}_{1}_{2}.reprojection_dict".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(reprojection_dict_file_path, "wb") as f:
        pickle.dump(temp_reprojection_dict, f)


def connect_contour(contour_list):
    con_list = []
    for col in contour_list:
        if len(col) == 0:
            con_list.append(np.empty((1, 2)))
            continue
        A = np.concatenate(col).reshape((-1, 2))
        con_list.append(A)
    return con_list


def distance_check(distance_list):
    """_summary_

    Parameters
    ----------
    distance_list : list of list of shape (n_labels, n_pairs, )
        _description_

    Returns
    -------
    _type_
        _description_

    TODO: distance_listの構造, 条件の引数化
    """
    dist_check_list = []
    ac_list = []
    for col in distance_list:
        col_list = []
        ac_col_list = []
        for frag in col:
            ac = np.array((np.min(frag, axis=1)) < 10)  # 条件:10 pixel以内
            col_list.append(np.sum(ac, dtype=np.uint16) / len(ac))
            ac_col_list.append(ac)
        ac_list.append(ac_col_list)
        dist_check_list.append(np.array(col_list))
    return ac_list, dist_check_list


def repro_sparse(repro_dict_taged):

    r_dict = {}
    for key in repro_dict_taged:
        k_list = []
        for label in repro_dict_taged[key]:
            l_list = []
            for frag in label:
                # COOで置き換え
                # a = coo_matrix((1080,1920),dtype=np.int16)
                n_frag, idx = np.unique(frag, axis=0, return_inverse=True)
                n_frag_len = len(n_frag)
                data = np.arange(n_frag_len) + 1
                col = n_frag[:, 0]
                row = n_frag[:, 1]

                # for i, coord in enumerate(n_frag):
                #    if (coord[1] < 1080) & (coord[1] >= 0) & (coord[0] < 1920) & (coord[0] >= 0):
                #        a[coord[1],coord[0]] = i+1

                if (
                    (np.sum(row >= 1080))
                    | (np.sum(row < 0))
                    | (np.sum(col >= 1920))
                    | (np.sum(col < 0))
                ):
                    l_list.append((0, idx, n_frag_len))


                else:
                    a = coo_matrix((data, (row, col)), shape=(1080, 1920))
                    l_list.append((a.tocsr(copy=True), idx, n_frag_len))

            k_list.append(l_list)
        r_dict[key] = k_list
    return r_dict


def cal_distance(repro_sparse_P, contour_sparse_P):
    dist_check_list = []
    ac_list = []
    for repro_col, con_col in zip(repro_sparse_P, contour_sparse_P):
        col_list = []
        ac_col_list = []
        for repro_frag in repro_col:
            if type(repro_frag[0]) == int:
                col_list.append(0)
                frag_ac = np.zeros(len(repro_frag[1]))
                ac_col_list.append(frag_ac)

            else:
                supported = 0
                frag_ac = np.zeros(len(repro_frag[1]))
                frag_len = repro_frag[0].count_nonzero()
                m_csr = repro_frag[0].multiply(con_col)
                partation = m_csr.count_nonzero() / frag_len

                if partation > 0.8:
                    supported = 1
                    supported_num = np.copy(m_csr.data)
                    frag_ac = np.zeros(repro_frag[2])
                    frag_ac[supported_num - 1] = 1
                    frag_ac = frag_ac[repro_frag[1]]
                col_list.append(supported)
                ac_col_list.append(frag_ac)
        dist_check_list.append(np.array(col_list))
        ac_list.append(ac_col_list)

    return ac_list, dist_check_list


def P_dict_check(repro_dict_taged, cam_list=[]):

    """_summary_

    Parameters
    ----------
    repro_dict_taged : _type_
        _description_
    cam_list : list, optional
        _description_, by default []

    Returns
    -------
    _type_
        _description_

    repro_dict_taged: reprojection_dictと同じ構造

    TODO: docstring
    """
    repro_dict_taged = repro_sparse(repro_dict_taged)
    P_list = []
    P_ac_list = []
    for P_tag in repro_dict_taged:
        repro_P = repro_dict_taged[P_tag]
        # contour_P = cam_list[P_tag].contour_list
        contour_P = cam_list[P_tag].contour_img
        # distance_list = cal_distance(repro_P, contour_P)
        # ac_list, dist_check_list = distance_check(distance_list)
        ac_list, dist_check_list = cal_distance(repro_P, contour_P)
        P_list.append(dist_check_list)
        P_ac_list.append(ac_list)
    P_check = np.array(P_list)
    return ac_list, P_ac_list, P_check


def P_check_integration(P_check):

    """_summary_

    Parameters
    ----------
    P_check : list of list of list of shape (n_cameras, n_labels, n_fragments_3d/n_pairs)
        n_cameras: 再投影先のカメラの数（全体からペアを除いたもの）

        曲線中でサポートを受けている割合（サポートを受けた交点の数/n_intersections)

    Returns
    -------
    check_list: list of list of shape (n_labels, n_fragments_3d/n_pairs)
        サポートの割合に対して閾値以上の再投影先の個数

    TODO: docstring, 閾値の引数化
    """
    check_list = []
    for col in range(P_check.shape[1]):
        temp_list = []
        for img in P_check[:, col]:
            temp = np.array(img > 0.8)  # 曲線中の何割が閾値以内か
            temp_list.append(img)
        col_check = np.sum(np.array(temp_list), axis=0, dtype=np.uint16)
        check_list.append(col_check)
    # print(check_list)
    return check_list


def ac_list_integration(P_ac_list):
    """交点毎のサポート

    Parameters
    ----------
    P_ac_list : list of list of list of shape (n_cameras, n_labels, n_fragments_3d/n_pairs, n_intersections) of bool
        n_cameras: 再投影先のカメラの数（全体からペアを除いたもの）

        交点毎でサポートを受けているかどうか

    Returns
    -------
    inter_ac: list of list of shape (n_labels, n_fragments_3d/n_pairs, n_intersections)
        サポートを受けた回数

    TODO: docstring
    """
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
    """サポートの計算
    Parameters
    ===================
    reprojection_dict: dict,
        key: tuple, camera_pair ((i,j),"F" or "R")
        val: dict,
            key: int, cam_num
            val: list, fragments

    Returns
    ========================
    support_dict: dict,
        key: tuple, camera_pair ((i,j),"F" or "R")
        val: tuple, (check_list, inter_ac)
        check_list[color][frag][support_num]: サポートの数を保存している
        inter_ac[color][frag][coodinate][support_num]: フラグメント上のどこがサポートを受けているか保存している
    """
    support_dict = {}
    for tag in reprojection_dict:
        repro_dict_taged = reprojection_dict[tag]
        _, P_ac_list, P_check = P_dict_check(repro_dict_taged, cam_list=cam_list)
        check_list = P_check_integration(P_check)
        inter_ac = ac_list_integration(P_ac_list)
        support_dict[tag] = (check_list, inter_ac)
    return support_dict


def gen_support(tag, cam_list=[]):
    """_summary_

    Parameters
    ----------
    tag : _type_
        _description_
    cam_list : list, optional
        _description_, by default []

    Returns
    -------
    _type_
        _description_

    X_Y_Z.support_dict: tuple of shape (2, )
        X: cam1のid
        Y: cam2のid
        Z: F or R
        tupleの各要素
            val1: list of shape (n_labels) of ndarray of shape (n_pairs, )
                あるペアのサポートの数
            val2: list of list of shape (n_label, n_pairs) of ndarray of shape (n_intersections)
                ある交点がサポートを受けた数

    """
    # if tag[1] == "R":
    #    return

    with open(
        r"temp/{0}_{1}_{2}.reprojection_dict".format(tag[0][0], tag[0][1], tag[1]), "rb"
    ) as f:
        repro_dict_taged = pickle.load(f)
    _, P_ac_list, P_check = P_dict_check(repro_dict_taged, cam_list=cam_list)
    check_list = P_check_integration(P_check)
    inter_ac = ac_list_integration(P_ac_list)
    os.remove(
        r"temp/{0}_{1}_{2}.reprojection_dict".format(tag[0][0], tag[0][1], tag[1])
    )
    with open(
        r"temp/{0}_{1}_{2}.support_dict".format(tag[0][0], tag[0][1], tag[1]), "wb"
    ) as f:

        pickle.dump((check_list, inter_ac), f)

    return  # support_dict
