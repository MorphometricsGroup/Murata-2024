##カメラペアを作る
'''Python
cam_list = [Camera(i) for i in range(48)]
for i in range(len(cam_list)):
    cam_list[i].img_load("../simlation_data/images/no_hole_leaf_image")
    cam_list[i].contour_extraction()
    cam_list[i].para_load("../simlation_data/view_mats/view_mat/{}.csv".format(str(i)))
    cam_list[i].get_contour_img()

cam_pairs = camera_correspondence(cam_list,angle_upper=40 / 180 * np.pi)
print(len(cam_pairs))
'''


##エピポール取得
'''Python
epipole_dict = {i:[] for i in range(len(cam_list))}
cam_pairs_F = {}
for i in cam_pairs:
    epipole1, epipole2, F = FF_mat(cam_list[i[0]].A, cam_list[i[1]].A, cam_list[i[0]].Rt, cam_list[i[1]].Rt)
    epipole_dict[i[0]].append(normalization(epipole1))
    epipole_dict[i[1]].append(normalization(epipole2))
    cam_pairs_F[i] = F
'''


##エピポーラ線に接する部分を削除する，輪郭をフラグメントにする．
'''Python
for i in range(len(cam_list)):
    im_del_list = all_D(epipole_angle(i, epipole_dict, cam_list=cam_list))# im_del_list[color][contour][del_idx]
    newCon = all_sep(cam_list[i].contour_list, im_del_list)# newCon[color][fragment][coordination]
    cam_list[i].frag_list = frag_list_fraged(newCon)

del epipole_dict
'''


##カーブフラグメントのペアを作る
'''Python
_ = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(pair_and_key_gen_one_label)(
    label_num, i, cam_list, cam_pairs_F) for i in cam_pairs_F)
'''


##カメラペアタグを取得
'''Python
temp_path = pathlib.Path("temp/")
tags_path = list(temp_path.glob("*.pair_list"))

tags = []
for tag_path in tags_path:
    tag_sp = tag_path.stem.split("_")
    tag_arr = ((int(tag_sp[0]), int(tag_sp[1])), tag_sp[2])
    tags.append(tag_arr)
'''


##カーブフラグメントペア内の点の対応を取得する
'''Python
_ = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(coll_dict_gen_one_label)(
    label_num, i, cam_list, cam_pairs_F) for i in tags)

_ = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(pair_pt_gen_one_label)(i) for i in tags)
'''


##インデックスから座標を引っ張ってくる
'''Python
_ = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(coordinate_dict_gen_one_label)(
    label_num, i, cam_list) for i in tags)
'''


##座標値から三次元再構築を行う
'''Python
_ = joblib.Parallel(n_jobs=-1,verbose=0)(joblib.delayed(TDlines_gen_one_label)(
    tag, cam_list=cam_list, cam_pairs_F=cam_pairs_F) for tag in tags)
'''


##各画像に再投影する
'''Python
_ = joblib.Parallel(n_jobs=-1,verbose=0)(joblib.delayed(reprojection_gen_one_label)(tag, cam_list=cam_list) for tag in tags)
'''


##再投影からサポートを計算する
'''Python
_ = joblib.Parallel(n_jobs=-1,verbose=0)(joblib.delayed(gen_support_one_label)(label_num, tag, cam_list, image_shape=(1080, 1920)) for tag in tags)
'''


##閾値以上のサポートを受けた3次元曲線フラグメントだけを取り出す
'''Python
sup_th = 20# サポート数
curve_fragment = []
for tag in tags:
    if tag[1] == "R":
        continue
    
    with open(r"temp/{0}_{1}_{2}.TDlines".format(tag[0][0],tag[0][1],tag[1]), 'rb') as f:
        lines_list = pickle.load(f)
    #lines_list = TDlines[tag]
        
    with open(r"temp/{0}_{1}_{2}.support_dict".format(tag[0][0],tag[0][1],tag[1]), 'rb') as f:
        support_list, support_ac = pickle.load(f)
    #support_list, support_ac = support_dict[tag][0], support_dict[tag][1]
    tag_list = []
    for frag, sup, sup_ac in zip(lines_list, support_list, support_ac):
        if sup > sup_th:
            frag = np.reshape(frag,(-1, 3))
            frag = np.array([i for i,j in zip(frag, sup_ac >sup_th) if j])
            tag_list.append(frag)
    curve_fragment.append(tag_list)
'''
