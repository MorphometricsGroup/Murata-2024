#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib

import numpy as np
import cv2


class Camera:
    def __init__(self, img_num, f=8000 / 3, cx=1920 / 2, cy=1080 / 2):
        self.img_num = img_num  # カメラ番号（int;コンストラクタ）

        A = np.zeros((3, 3))
        A[0, 0] = f
        A[0, 2] = cx
        A[1, 1] = f
        A[1, 2] = cy
        A[2, 2] = 1

        self.A = A  # 内部パラメータ(ndarray)後から更新

    def img_load(self, dir_path="images/one_hole_plant_image"):
        # folder_path = "images/one_hole_plant_image"
        file_path = os.path.join(dir_path, str(self.img_num) + ".png")
        img = cv2.imread(file_path, 1)  # BGRで読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        self.img = img  # 画像(ndarray)

    def contour_extraction(self):

        color_arr = np.array(
            [
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
            dtype=np.int16,
        )
        masks = np.ones((self.img.shape[0], self.img.shape[1], 9), dtype=np.uint8)

        for i, color in enumerate(color_arr):
            lower = np.clip(color, 0, 255)
            upper = np.clip(color, 0, 255)
            img_mask = cv2.inRange(self.img, lower, upper)
            masks[:, :, i] = img_mask

        # self.masks = masks # 色ごとのマスク(nd.array)

        contour_list = []

        # 色ごとに輪郭（閉曲線）を抽出
        for i in range(masks.shape[2]):
            contours, hierarchy = cv2.findContours(
                masks[:, :, i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            contour_list.append(contours)
        self.contour_list = contour_list  # 輪郭のリスト(list,ndarray)

        # self.frag_list = contours2fragments(self.contour_list) # フラグメントのリスト(list,ndarray)

    def para_load(self, dir_path=pathlib.Path("view_mats/view_mat")):

        # folder_path = pathlib.Path("view_mats/view_mat")
        file_path = os.path.join(dir_path, str(self.img_num) + ".csv")
        self.Rt = np.loadtxt(file_path, delimiter="\t")
        self.P = np.dot(self.A, self.Rt[0:3, 0:4])
        self.cam_world_cood = -np.dot(self.Rt[0:3, 0:3].T, self.Rt[0:3, 3])
