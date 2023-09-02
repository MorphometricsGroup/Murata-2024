import numpy as np
import math
import bpy
import copy

rng = np.random.default_rng()

def rotation_mat(angle):
    Rx = np.array([[1,0,0],
                 [0, np.cos(angle[0]), -np.sin(angle[0])],
                 [0, np.sin(angle[0]), np.cos(angle[0])]])
    Ry = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])],
                 [0,1,0],
                 [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    Rz = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                 [np.sin(angle[2]), np.cos(angle[2]), 0],
                 [0,0,1]])
    return Rz@Rx@Ry

def duplicate_object_rename(arg_objectname='Default', arg_dupname=''):
   for ob in bpy.context.scene.objects:
     ob.select_set(False)
   selectob=bpy.context.scene.objects[arg_objectname]
   selectob.select_set(True)
   bpy.ops.object.duplicate_move(
    OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
   selectob.select_set(False)
   if len(arg_dupname) > 0:
     duplicated_objectname=arg_objectname + ".001"
     duplicatedob=bpy.data.objects[duplicated_objectname]
     duplicatedob.name=arg_dupname
   return

def get_rept_core(vers):
    temp_rept = np.zeros(3)
    for ver in vers:
        coor = np.asarray(ver.co)
        temp_rept += coor
    rept = temp_rept/len(vers)
    return rept

def get_rept(msh):
    rept_arr = np.zeros((len(msh), 3))
    for i in range(len(msh)):
        vers = msh[i].vertices
        rept = get_rept_core(vers)
        rept_arr[i] = rept
    return rept_arr

def get_max_min_ver():
    vers_list = []
    for ob in bpy.context.scene.objects:
        msh=ob.data
        vers = msh.vertices
        vers_list += [np.asarray(ver.co) for ver in vers]
    vers_list = np.stack(vers_list)
    print(vers_list)
    ver_max = np.max(vers_list, axis=0)
    ver_min = np.min(vers_list, axis=0)
    return ver_max, ver_min
    
def make_bbox(max_arr, min_arr):
    bbox = np.array([
        (max_arr[0],max_arr[2],max_arr[1]),
        (max_arr[0],max_arr[2],min_arr[1]),
        (max_arr[0],min_arr[2],min_arr[1]),
        (max_arr[0],min_arr[2],max_arr[1]),
        (min_arr[0],max_arr[2],max_arr[1]),
        (min_arr[0],max_arr[2],min_arr[1]),
        (min_arr[0],min_arr[2],min_arr[1]),
        (min_arr[0],min_arr[2],max_arr[1])
    ])
    faces = [(0,1,2,3),(0,1,5,4),(0,3,7,4),(1,2,6,5),(2,3,7,6),(4,5,6,7)]
    # 2,3,7,6が底面
    return bbox, faces

def point_on_suf(start, end1, end2):
    s = end1 - start
    t = end2 - start
    return s * 1 + t * 1 + start

def gen_o_leaf_pos(leaf_num, bbox, faces):
    face_candidate = [0,1,2,3,5]
    o_leaf_pos_list = []
    for i in range(leaf_num):
        face_candidate_idx = rng.integers(0, len(face_candidate))
        face_idx = face_candidate[face_candidate_idx]
        face = faces[face_idx]
        face_ver = [bbox[face_num] for face_num in face]
        o_leaf_pos = point_on_suf(face_ver[0], face_ver[1], face_ver[3])
        o_leaf_pos_list.append(o_leaf_pos)
    return o_leaf_pos_list



leaf_num = 8

ver_max, ver_min = get_max_min_ver()
#ver_max += 2
#ver_min -= 2
#ver_min[2] += 4
max_arr, min_arr = ver_max, ver_min
bbox, faces = make_bbox(max_arr, min_arr)
print(bbox)
o_leaf_pos_list = gen_o_leaf_pos(leaf_num, bbox, faces)
a_r = np.ones(3) * math.radians(-10)
b_r = np.ones(3) * math.radians(10)

for i in range(8):
    o_name = "o_leaf_{}".format(i)
    duplicate_object_rename(arg_objectname='leaf_0', arg_dupname=o_name)

s = 0
for ob in bpy.context.scene.objects:
    msh=ob.data
    if "o_leaf" in ob.name:
        vers = msh.vertices
        rot = ob.rotation_euler
        rept = get_rept_core(vers)
        diff = o_leaf_pos_list[s]
        diff_rot = (b_r - a_r) * rng.random()  + a_r
        for j in range(len(vers)):
            rot_ori = np.asarray(vers[j].co) - rept
            rot_co = rotation_mat(diff_rot)@rot_ori
            vers[j].co = rot_co + diff
        s += 1