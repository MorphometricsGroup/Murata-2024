#! /usr/bin/env python
# -*- coding: utf-8 -*-


def normalization(vec3):
    """同次座標への変換を与える

    Todo: 名前をもう少し特殊なものに変えたい．
    """
    return vec3[0] / vec3[2], vec3[1] / vec3[2]
