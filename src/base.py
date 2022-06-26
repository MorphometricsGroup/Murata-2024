#! /usr/bin/env python
# -*- coding: utf-8 -*-


def normalization(vec3):
    """
    Todo: 名前をもう少し特殊なものに変えたい．
    """
    return vec3[0] / vec3[2], vec3[1] / vec3[2]
