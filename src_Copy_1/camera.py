#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import pickle

import numpy as np
import cv2
from scipy.sparse import csr_matrix


class Camera:
    def __init__(self, img_num, f=8000 / 3, cx=1920 / 2, cy=1080 / 2):
        self.img_num = img_num  # カメラ番号（int;コンストラクタ）

        A = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        self.A = A  # 内部パラメータ(ndarray)後から更新

    def img_load(self, dir_path):
        """_summary_

        Args:
            file_path (_type_): _description_

        TODO: 画像を1チャネルに変える
        """
        file_path = os.path.join(dir_path, str(self.img_num) + ".png")
        img = cv2.imread(file_path, 1)  # BGRで読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        self.img = img  # 画像(ndarray)

    def contour_extraction(
        self,
        labels=[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [127, 127, 127],
            [127, 0, 127],
            [0, 127, 127],
        ],
    ):
        """Extract contours based on their labels (colors)

        Args:
            labels (_type_): _description_

        TODO: 画像を1チャネルに変える
        """

        n_labels = len(labels)
        color_arr = np.array(labels, dtype=np.int16)
        masks = np.ones(
            (self.img.shape[0], self.img.shape[1], n_labels), dtype=np.uint8
        )

        for i, color in enumerate(color_arr):
            lower = np.clip(color, 0, 255)
            upper = np.clip(color, 0, 255)
            img_mask = cv2.inRange(self.img, lower, upper)
            masks[:, :, i] = img_mask

        self.masks = masks # 色ごとのマスク(nd.array)

        contour_list = []

        # 色ごとに輪郭（閉曲線）を抽出
        for i in range(masks.shape[2]):
            contours, hierarchy = cv2.findContours(
                masks[:, :, i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            contour_list.append(contours)
        self.contour_list = contour_list  # 輪郭のリスト(list,ndarray)

        # self.frag_list = contours2fragments(self.contour_list) # フラグメントのリスト(list,ndarray)

    def para_load(self, file_path):
        self.Rt = np.loadtxt(file_path, delimiter="\t")
        self.P = np.dot(self.A, self.Rt[0:3, 0:4])
        self.cam_world_cood = -np.dot(self.Rt[0:3, 0:3].T, self.Rt[0:3, 3])
        
    def get_contour_img(self):
        kernel = np.ones((3,3),np.uint8)
        sparse_mats_list = []
        for j in range(len(self.contour_list)):
            new_img = np.zeros((self.img.shape[0], self.img.shape[1]),dtype=np.uint8)
            for i in range(len(self.contour_list[j])):
                curve = self.contour_list[j][i][~np.isnan(self.contour_list[j][i])].reshape((-1,2)).astype(int)
                new_img[curve[:,1],curve[:,0]]=True
            dilation = cv2.dilate(new_img, kernel, iterations = 10)
            new_img = csr_matrix(dilation, dtype=np.uint8)
            sparse_mats_list.append(new_img)

        self.contour_img = sparse_mats_list


def cood_to_mask(csv_path, im_shape):
    idx = np.loadtxt(str(csv_path), delimiter=",")
    idx = idx.astype(np.int64)

    mask = np.zeros(im_shape, dtype=np.uint8)

    if idx.size == 0:
        return mask

    mask[idx[0], idx[1]] = 255
    return mask


class metashape_Camera:
    def __init__(self, val, inner_para_path="view_mat/inner_para.csv"):
        self.img_num = val
        inner_para_path_ = pathlib.Path(inner_para_path)
        self.A = np.loadtxt(str(inner_para_path_), delimiter=",")

    def img_load(self, folder_path="GmJMC025_02"):
        file_path = os.path.join(folder_path, str(self.img_num) + ".JPG")
        img = cv2.imread(file_path, 1)  # BGRで読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.flip(img, 1) # 反転 unity用?
        self.img = img  # 画像(ndarray)
        self.img_shape = img.shape

    def contour_load(self, folder="masks/GmJMC025_02_mask"):
        folder_ = pathlib.Path(folder)
        masks_path = folder_.glob(str(self.img_num) + "_" + "*" + ".csv")

        # mask_list = []
        contour_list = []
        for mask_path in masks_path:
            mask = cood_to_mask(mask_path, (self.img_shape[0], self.img_shape[1]))
            #mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)            
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            # mask_list.append(mask)
            contour_list.append(contours)
        # self.masks = np.asarray(mask_list).transpose(1, 2, 0)
        self.contour_list = contour_list

    def label_load(self, label_path="genelated_mask_label.pickle"):
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        max_num = 0
        for label in self.labels:
            temp_max = np.max(label)
            if max_num < temp_max:
                max_num = temp_max
        self.max_num = max_num

    def correspondence_contour(self):
        correspondence_list = []

        for i in range(self.max_num + 1):
            temp_c_list = []
            idx = np.where(self.labels[self.img_num] == i)
            if idx[0].size != 0:
                for j in idx[0]:
                    temp_c_list += self.contour_list[j]
            correspondence_list.append(temp_c_list)
        self.contour_list = correspondence_list

    def para_load(self, folder_path="view_mat"):
        folder_path_ = pathlib.Path(folder_path)
        file_path = os.path.join(folder_path_, str(self.img_num) + ".csv")
        self.Rt = np.loadtxt(file_path, delimiter=",")
        self.Rt = np.linalg.pinv(self.Rt)
        self.P = np.dot(self.A, self.Rt[0:3, 0:4])
        self.cam_world_cood = -np.dot(self.Rt[0:3, 0:3].T, self.Rt[0:3, 3])


def cam_pos_mean(cam_list):
    _cam_pos = np.zeros(3)
    for cam in cam_list:
        _cam_pos += cam.cam_world_cood
    cam_mean = _cam_pos / len(cam_list)
    return cam_mean


def vec_L2(vec):
    """
    ToDo: numpy.linalg.normで置き換える？
    """
    return np.sum(vec**2) ** (1 / 2)


def cal_angle(cam_pos1, cam_pos2, cam_mean):
    vec1 = cam_pos1 - cam_mean
    vec2 = cam_pos2 - cam_mean
    cossin = np.dot(vec1, vec2) / (vec_L2(vec1) * vec_L2(vec2))
    angle = np.arccos(cossin)
    return angle


def camera_correspondence(cam_list, angle_upper=1 / 9 * np.pi):
    """カメラの対応を返す
    Parameters
    ========================
    cam_list: cameraのlist
    angle_upper: エピポーラ角の上限（論文では20度）

    Returns
    ========================
    pair_list: list of tuple (i,j), i, j: camera_number

    TODO: i,jとimg_num（カメラ番号）のいずれかを使う

    """

    pair_list = []
    cam_mean = cam_pos_mean(cam_list)
    for i, cam1 in enumerate(cam_list):
        for j, cam2 in enumerate(cam_list):
            if i == j or i > j:
                continue
            cam1_pos = cam1.cam_world_cood
            cam2_pos = cam2.cam_world_cood
            angle = cal_angle(cam1_pos, cam2_pos, cam_mean)
            if angle < angle_upper:
                pair_list.append((i, j))
    return pair_list
