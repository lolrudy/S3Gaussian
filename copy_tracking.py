import os
import shutil

view0_tracking_dir = '/home/rudy/下载/updated_view_0/final_results_view_0'
view1_tracking_dir = '/home/rudy/下载/updated_view_1/updated_pkl_view_1'
view2_tracking_dir = '/home/rudy/下载/updated_view_2/exp_view_2_kf_new3'

data_dir = 'data/waymo/processed/dynamic/training'
# data_dir = 'data/waymo/processed/static/training'

scene_seg_mapping = {
    16: "seg102319",
    21: "seg103913",
    22: "seg104444",
    25: "seg104980",
    31: "seg105887",
    34: "seg106250",
    35: "seg106648",
    49: "seg109636",
    53: "seg110170",
    80: "seg117188",
    84: "seg118463",
    86: "seg119178",
    89: "seg119284",
    94: "seg120278",
    96: "seg121618",
    102: "seg122514",
    111: "seg123392",
    222: "seg148106",
    323: "seg168016",
    382: "seg181118",
    402: "seg191876",
    427: "seg225932",
    438: "seg254789",
    546: "seg441423",
    581: "seg508351",
    592: "seg522233",
    620: "seg583504",
    640: "seg624282",
    700: "seg767010",
    754: "seg882250",
    795: "seg990779",
    796: "seg990914",
    3: "seg100613",
    19: "seg102751",
    36: "seg106762",
    69: "seg113792",
    81: "seg117240",
    126: "seg128796",
    139: "seg130854",
    140: "seg131421",
    146: "seg131967",
    148: "seg132384",
    157: "seg134763",
    181: "seg140045",
    200: "seg143481",
    204: "seg144248",
    226: "seg148697",
    232: "seg150623",
    237: "seg152217",
    241: "seg152706",
    245: "seg153495",
    246: "seg153658",
    271: "seg158686",
    297: "seg163453",
    302: "seg164701",
    312: "seg166085",
    314: "seg166463",
    362: "seg177619",
    482: "seg322492",
    495: "seg342571",
    524: "seg398895",
    527: "seg405841",
    753: "seg881121",
    780: "seg938501"
}

scene_list = os.listdir(data_dir)

view0_list = os.listdir(view0_tracking_dir)
view1_list = os.listdir(view1_tracking_dir)
view2_list = os.listdir(view2_tracking_dir)

for scene_name in scene_list:
    curr_track_dir = os.path.join(data_dir, scene_name, 'tracking')
    os.makedirs(curr_track_dir, exist_ok=True)
    seg_name = scene_seg_mapping[int(scene_name)][-6:]
    print(scene_name, seg_name)
    view0_pkl_path = [i for i in view0_list if seg_name in i and 'pkl' in i]
    view1_pkl_path = [i for i in view1_list if seg_name in i and 'pkl' in i]
    view2_pkl_path = [i for i in view2_list if seg_name in i and 'pkl' in i]
    if len(view0_pkl_path) > 1:
        view0_pkl_path = [i for i in view0_pkl_path if 'modified' in i]
        assert len(view0_pkl_path) == 1
    view0_pkl_path = view0_pkl_path[0]
    if len(view1_pkl_path) > 1:
        view1_pkl_path = [i for i in view1_pkl_path if 'modified' in i]
        assert len(view1_pkl_path) == 1
    view1_pkl_path = view1_pkl_path[0]
    if len(view2_pkl_path) > 1:
        view2_pkl_path = [i for i in view2_pkl_path if 'modified' in i]
        assert len(view2_pkl_path) == 1
    view2_pkl_path = view2_pkl_path[0]

    shutil.copy2(os.path.join(view0_tracking_dir, view0_pkl_path), os.path.join(curr_track_dir, 'view_0.pkl'))
    shutil.copy2(os.path.join(view1_tracking_dir, view1_pkl_path), os.path.join(curr_track_dir, 'view_1.pkl'))
    shutil.copy2(os.path.join(view2_tracking_dir, view2_pkl_path), os.path.join(curr_track_dir, 'view_2.pkl'))