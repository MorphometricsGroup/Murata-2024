#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pickle
import os


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
    """エピポールと基礎行列Fを返す
    Parameters
    ========================
    A1, A2: 3*3 array, inner parameter
    Rt1, Rt2: 3*4 array, external parameter

    Returns
    ========================
    epipole1, epipole2: 3 array
    F matrix: 3*3 array

    """

    P1 = np.dot(A1, Rt1[0:3, 0:4])
    P2 = np.dot(A2, Rt2[0:3, 0:4])
    cam_pos1 = -np.dot(Rt1[0:3, 0:3].T, Rt1[0:3, 3])
    cam_pos1 = np.array([cam_pos1[0], cam_pos1[1], cam_pos1[2], 1])
    epipole2 = np.dot(P2, cam_pos1)
    cam_pos2 = -np.dot(Rt2[0:3, 0:3].T, Rt2[0:3, 3])
    cam_pos2 = np.array([cam_pos2[0], cam_pos2[1], cam_pos2[2], 1])
    epipole1 = np.dot(P1, cam_pos2)

    Fmat = np.dot(SS_mat(epipole2), np.dot(P2, np.linalg.pinv(P1)))
    return epipole1, epipole2, Fmat


def gene(angles):
    """
    TODO: fragmentと重複している
    """
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
    # cam.img_load()
    # cam.contour_extraction()
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


def epilines_para_one_label(frags, F, label_num):

    para_list_labeled = []
    frag_labeled = frags[label_num]
    for frag in frag_labeled:
        frag_lines = cv2.computeCorrespondEpilines(
            frag.reshape(-1, 1, 2), 1, F
        )
        para_list_labeled.append(frag_lines)

    return para_list_labeled


def get_frag_cood_one_label(frag_list, label_num):
    cood_S_labeled = []
    cood_F_labeled = []
    frag_labeled = frag_list[label_num]
    col_S = np.array([frag[0] for frag in frag_labeled])
    col_F = np.array([frag[-1] for frag in frag_labeled])
    cood_S_labeled.append(col_S)
    cood_F_labeled.append(col_F)
    return cood_S_labeled, cood_F_labeled  # cood_S[color][frag]


def para2cood_S(para_list):
    return np.array([[0, -c / b] for a, b, c in para_list])


def para2cood_F(para_list, img_width=1920):
    return np.array([[img_width, -(img_width * a + c) / b] for a, b, c in para_list])


def all_pa2co_one_label(para_list_labeled):
    epi_cood_S_labeled = []
    epi_cood_F_labeled = []
    for frag in para_list_labeled:
        S_cood = para2cood_S(frag.squeeze())
        F_cood = para2cood_F(frag.squeeze())
        epi_cood_S_labeled.append(S_cood)
        epi_cood_F_labeled.append(F_cood)
    return epi_cood_S_labeled, epi_cood_F_labeled


def coll_t1_t2(epi_cood_S, epi_cood_F, cood_S, cood_F):
    epi_cood_S_bro = np.repeat(epi_cood_S, len(cood_S), axis=0).reshape(
        (epi_cood_S.shape[0], len(cood_S), epi_cood_S.shape[1])
    )
    epi_cood_F_bro = np.repeat(epi_cood_F, len(cood_S), axis=0).reshape(
        (epi_cood_F.shape[0], len(cood_S), epi_cood_F.shape[1])
    )
    v = cood_S - epi_cood_S_bro
    v2 = np.array(cood_F) - np.array(cood_S)
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


def make_pair_list_one_label(epi_cood_S_labeled, epi_cood_F_labeled, cood_S_labeled, cood_F_labeled):
    pair_list_labeled = []
    if len(epi_cood_S_labeled) == 0 or len(cood_S_labeled) == 0:
        return pair_list_labeled
        
    for epi_S_frag, epi_F_frag in zip(epi_cood_S_labeled, epi_cood_F_labeled):
        t1, t2 = coll_t1_t2(epi_S_frag, epi_F_frag, cood_S_labeled, cood_F_labeled)
        surport_idx = coll_det(t1, t2)
        pair_list_labeled.append(surport_idx)
    return pair_list_labeled


def pair_and_key_gen_one_label(label_num, pair, cam_list=[], cam_pairs_F=[], dest_dir="temp"):
    """
    Parameters
    ======================
    pair: tuple of int, of shape (2, )
        カメラのペアを指定するtuple

    Returns
    ======================
    pair_list: dict, key: ( pair, 'F' or 'R') , val:
        曲線のペアのリスト
        ある画像中のfragmentsに対して，もう一方の画像で対応するfragmentsのindex

    Notes
    =========================
    'F'と'R': forward（i から j）, reverse（jからi）

    X_Y_Z.pair_list: list of list of ndarray, of shape (n_labels, n_fragments_cam1, n_fragment_ids_cam2)
        X: cam1のid
        Y: cam2のid
        Z: F or R
        n_labels: ラベルの数
        n_fragments_cam1: cam1内にあるfragmentsの数
        n_fragment_ids_cam2: cam1内のあるfragmentに対応するcam2内のfragmentのidの数
    """
    #pair_list = {}
    F = cam_pairs_F[pair]
    frags_para12_one_label = epilines_para_one_label(cam_list[pair[0]].frag_list, F, label_num)  # frags_para[色][frag]

    cood_S_one_label, cood_F_one_label = get_frag_cood_one_label(cam_list[pair[1]].frag_list, label_num)
    epi_cood_S_one_label, epi_cood_F_one_label = all_pa2co_one_label(frags_para12_one_label)
    img_list1 = make_pair_list_one_label(epi_cood_S_one_label, epi_cood_F_one_label, cood_S_one_label, cood_F_one_label)

    # cood_S, cood_F = get_frag_cood(cam_list[pair[0]].frag_list)
    # epi_cood_S, epi_cood_F = all_pa2co(frags_para21)
    # img_list2 = make_pair_list(epi_cood_S, epi_cood_F, cood_S, cood_F)

    # pair_list[(pair, "F")] = img_list1
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_file_path = os.path.join(
        dest_dir, r"{0}_{1}_{2}.pair_list".format(pair[0], pair[1], "F")
    )
    with open(dest_file_path, "wb") as f:
        pickle.dump(img_list1, f)

    # pair_list[(pair, "R")] = img_list2
    return 


def PL_coll_one_label(label_num, pair, pair_list_taged_labeled, cam_list, cam_pairs_F=[]):
    """線と点の衝突判定"""
    F = cam_pairs_F[pair[0]]
    if pair[1] == "F":
        frags_para_labeled = epilines_para_one_label(cam_list[pair[0][0]].frag_list, F, label_num)
        epi_cood_S_labeled, epi_cood_F_labeled = all_pa2co_one_label(frags_para_labeled)
        camL_idx = pair[0][1]
        
    elif pair[1] == "R":
        frags_para_labeled = epilines_para_one_label(cam_list[pair[0][1]].frag_list, F.T, label_num)
        epi_cood_S_labeled, epi_cood_F_labeled = all_pa2co_one_label(frags_para_labeled)
        camL_idx = pair[0][0]

    im_list = []
    flag_list_labeled = cam_list[camL_idx].frag_list[label_num]

    for pair_frag, epi_S_frag, epi_F_frag in zip(pair_list_taged_labeled, epi_cood_S_labeled, epi_cood_F_labeled):
        f_list = []
        for pair_frag_each in pair_frag:
            pts = flag_list_labeled[pair_frag_each]  # 対応するフラグメント
            v1 = epi_F_frag - epi_S_frag
            v1_n = (v1[:, 0] ** 2 + v1[:, 1] ** 2) ** (1 / 2)
            v1_n = np.stack([v1_n, v1_n], axis=1)
            v1 = v1 / v1_n
            v1_bro = np.repeat(v1, len(pts), axis=0).reshape(
                (v1.shape[0], len(pts), v1.shape[1])
            )
            epi_cood_S_bro = np.repeat(epi_S_frag, len(pts), axis=0).reshape(
                (epi_S_frag.shape[0], len(pts), epi_S_frag.shape[1])
            )

            v2 = pts - epi_cood_S_bro
            v2_n = (v2[:, :, 0] ** 2 + v2[:, :, 1] ** 2) ** (1 / 2)
            v2_n = np.stack([v2_n, v2_n], axis=2)
            v2 = v2 / v2_n
            con_det = np.cross(v1_bro, v2)
            f_list.append(np.where(np.abs(con_det) <= 0.001))
        im_list.append(f_list)

    return im_list

        
def coll_dict_gen_one_label(label_num, pair, cam_list=[], cam_pairs_F=[]):
    """点と線の衝突判定
    Parameters
    ======================
    pair: tuple of int
        カメラのペアとF，Rを指定するtuple

    Returns
    ======================
    coll_dict: dict,    key:  ( pair, 'F' or 'R') ,
                        val: カメラ1のある曲線上の指定された1点に対応する，カメラ2上のpair_listで指定された曲線上の複数のidx

    X_Y_Z.coll_dict: list of list of list of tuple of ndarray, of shape (n_labels, n_fragments_cam1, n_fragments_cam2, 2,n_pixel_ids_of_fragment_cam2)
        X: cam1のid
        Y: cam2のid
        Z: F or R
        n_labels: ラベルの数
        n_fragments_cam1: cam1内にあるfragmentsの数
        n_fragments_cam2: cam1内のあるfragmentに対応するcam2内のfragmentの数
        n_pixel_ids_of_fragment: cam1,cam2のfragmentの交わる画素idの数

        tuple: cam1側の画素id，cam2側の画素id

    """
    # coll_dict = {}
    with open(
        r"temp/{0}_{1}_{2}.pair_list".format(pair[0][0], pair[0][1], pair[1]), "rb"
    ) as f:
        pair_list_taged_labeled = pickle.load(f)

    im_list = PL_coll_one_label(label_num, pair, pair_list_taged_labeled, cam_list, cam_pairs_F)
    
    with open(
        r"temp/{0}_{1}_{2}.coll_dict".format(pair[0][0], pair[0][1], pair[1]), "wb"
    ) as f:
        pickle.dump(im_list, f)
        
        
def pt_pair(coll_list):
    """点と線の衝突判定
    Parameters
    ======================
    coll_list:

    Returns
    ======================
    list, col_dictで取った候補を1点対1点で対応させる

    """
    pool_i = []
    pool_j = []
    pre_i = None
    pre_j = None
    pt = 1
    for i, j in zip(coll_list[0], coll_list[1]):
        if i in pool_i:
            if pt == 1:
                continue
            elif pt == 0:
                if j not in pool_j:
                    pool_i.pop()
                    pool_j.pop()
                    pool_i.append(i)
                    pool_j.append(j)
                else:
                    continue

        elif i not in pool_i:
            if j in pool_j:
                pt = 0
            else:
                pt = 1
            pool_i.append(i)
            pool_j.append(j)
    return np.array([pool_i, pool_j])


def pair_pt_gen_one_label(tag):
    """_summary_

    Parameters
    ----------
    tag : _type_
        _description_

    X_Y_Z.pair_pt: list of list of list of tuple of ndarray, of shape (n_labels, n_fragments_cam1, n_fragments_cam2, 2,n_pixel_ids_of_fragment_cam2)
        X: cam1のid
        Y: cam2のid
        Z: F or R
        n_labels: ラベルの数
        n_fragments_cam1: cam1内にあるfragmentsの数
        n_fragments_cam2: cam1内のあるfragmentに対応するcam2内のfragmentの数
        n_pixel_ids_of_fragment: cam1,cam2のfragmentの交わる画素idの数

        coll_dictから交差を除去したもの

    """
    
    with open(
        r"temp/{0}_{1}_{2}.coll_dict".format(tag[0][0], tag[0][1], tag[1]), "rb"
    ) as f:
        coll_dict_taged_labeled = pickle.load(f)
    
    im_list = []
    for frag in coll_dict_taged_labeled:
        f_list = []
        for each_frag in frag:
            new_pair = pt_pair(each_frag)
            f_list.append(new_pair)    
        im_list.append(f_list)

        # pair_pt[i] = im_list
    os.remove(r"temp/{0}_{1}_{2}.coll_dict".format(tag[0][0], tag[0][1], tag[1]))

    with open(
        r"temp/{0}_{1}_{2}.pair_pt".format(tag[0][0], tag[0][1], tag[1]), "wb"
    ) as f:
        pickle.dump(im_list, f)