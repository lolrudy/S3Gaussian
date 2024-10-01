#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import os
import pickle
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud
from tqdm import trange
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import cv2
from utils.general_utils import sample_on_aabb_surface, get_OccGrid
from utils.segmentation_utils import get_panoptic_id
import torch
from utils.feature_extractor import extract_and_save_features
from utils.image_utils import get_robust_pca
from utils.refs import SEG_ID2NAME, SEG_NAME2ID, DYNAMIC_OBJECT_ID, STATIC_OBJECT_ID, VEHICLE_ID, FLAT_ID, THING
from pyrotation.conversion import matrix_from_quaternion
import pycocotools.mask as mask_util
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    # for waymo
    sky_mask: np.array = None
    depth_map: np.array = None
    time: float = None
    semantic_mask: np.array = None
    instance_mask: np.array = None
    sam_mask: np.array = None
    dynamic_mask: np.array = None
    feat_map: np.array = None
    # grouping
    objects: np.array = None
    # 
    intrinsic: np.array = None
    c2w: np.array = None
    gt_bboxes: list = None
    dynamic_mask_seman: np.array = None
    vehicle_points: np.array = None
    vehicle_colors: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    # for waymo
    full_cameras:list
    bg_point_cloud: BasicPointCloud = None
    ply_path: str = None
    bg_ply_path: str = None
    cam_frustum_aabb: np.array = None
    num_panoptic_objects: int = 0
    panoptic_id_to_idx: dict = None
    panoptic_object_ids: list = None
    occ_grid: np.array = None
    dynamic_point_cloud: BasicPointCloud = None
    road_pcd: BasicPointCloud = None
    sky_pcd: BasicPointCloud = None
    vehicle_pcd_dict: dict = None
    static_vehicle_pcd: BasicPointCloud = None
    vehicle_init_pose_dict: dict = None
    vehicle_pcd: BasicPointCloud = None


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) if os.path.exists(image_path) else None

        #object_path = os.path.join(objects_folder, image_name + '.png')
        #objects = Image.open(object_path) if os.path.exists(object_path) else None
        if 'test' not in image_name:
            # For Training, we use SAM-auto-mask
            if os.path.exists(os.path.join(objects_folder, image_name + '.png')):
                object_path = os.path.join(objects_folder, image_name + '.png')
                objects = Image.open(object_path)
            elif os.path.exists(os.path.join(objects_folder, image_name + '.jpg')):
                object_path = os.path.join(objects_folder, image_name + '.jpg')
                objects = Image.open(object_path)
            else:
                objects = None
        else:
            # For Testing, we use labeled-mask
            object_path = os.path.join(os.path.dirname(objects_folder),'object_mask', image_name + '.png')
            objects = Image.open(object_path) if os.path.exists(object_path) else None



        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    green_color = [0, 255, 0]  # [N,3] array
    rgb = np.array([green_color for _ in range(xyz.shape[0])])

    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, n_views=100, random_init=False, train_split=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    object_dir = 'object_mask' if object_path == None else object_path
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        ply_path = os.path.join(path, "sparse/0/points3D_randinit.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def constructCameras_waymo(frames_list, white_background, mapper = {},
                           load_intrinsic=False, load_c2w=False, start_time = 50, original_start_time = 0):
    cam_infos = []
    for idx, frame in enumerate(frames_list):
        # current frame time
        time = mapper[frame["time"]]
        # ------------------
        # load c2w
        # ------------------
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV/COLMAP (Y down, Z forward)
        #c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # ------------------
        # load image
        # ------------------
        cam_name = image_path = frame['file_path']
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0]) # d-nerf 透明背景
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        load_size = frame["load_size"]
        #image = PILtoTorch(image, load_size) #(800,800))
        # resize to load_size
        image = image.resize(load_size, Image.BILINEAR)
        # save pil image
        # image.save(os.path.join("debug", image_name + ".png"))
        # ------------------
        # load depth-map
        # ------------------
        depth_map = frame.get('depth_map', None)
        
        # # visualize depth map with rgb
        # mask = depth_map > 0
        # # normalize depth map to [0, 255]
        # depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        # np_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=1.0), cv2.COLORMAP_JET)
        # # mask empty depth map: depth_map(h,w) , np_depth_map(h,w,3)
        # np_depth_map[~mask] = [255, 255, 255]
        # depth_map_colored = Image.fromarray(np_depth_map)
        # #image_depth = Image.blend(image, depth_map_colored, 0.5)
        # image_np = np.array(image)
        # image_np[mask] = np_depth_map[mask]
        # image_depth = Image.fromarray(image_np)
        # image_depth.save(os.path.join("exp/debug-0", image_name + "_depth.png"))
        # depth_map_colored.save(os.path.join("exp/debug-0", image_name + "_depth_colored.png"))
        
        # ------------------
        # load sky-mask
        # ------------------
        sky_mask_path, sky_mask = frame["sky_mask_path"], None
        if sky_mask_path is not None:
            sky_mask = Image.open(sky_mask_path)
            sky_mask = sky_mask.resize(load_size, Image.BILINEAR)
        # ------------------
        # load intrinsic
        # ------------------
        # intrinsic to fov: intrinsic 已经被 scale
        intrinsic = frame["intrinsic"]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        # get fov
        fovx = focal2fov(fx, image.size[0])
        fovy = focal2fov(fy, image.size[1])
        FovY = fovy
        FovX = fovx

        semantic_mask = frame["semantic_mask"]
        instance_mask = frame["instance_mask"]
        dynamic_mask_seman = frame['dynamic_mask']

        # ------------------
        # load sam mask
        # ------------------
        sam_mask_path, sam_mask = frame["sam_mask_path"], None
        if sam_mask_path is not None:
            sam_mask = Image.open(sam_mask_path)
            sam_mask = sam_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #sam_mask = np.array(sam_mask) #.unsqueeze(-1)

        # ------------------
        # load dynamic mask
        # ------------------
        dynamic_mask_path, dynamic_mask = frame["dynamic_mask_path"], None
        if dynamic_mask_path is not None:
            dynamic_mask = Image.open(dynamic_mask_path)
            dynamic_mask = dynamic_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #dynamic_mask = np.array(dynamic_mask) #.unsqueeze(-1)

        # ------------------
        # NO load feat map
        # ------------------
        feat_map_path, feat_map = frame["feat_map_path"], None

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                        # for waymo
                        sky_mask=sky_mask, depth_map=depth_map, time=time,
                        semantic_mask=semantic_mask, instance_mask=instance_mask, 
                        sam_mask=sam_mask, 
                        dynamic_mask=dynamic_mask, 
                        feat_map=feat_map, # [640,960,3]
                        intrinsic=intrinsic if load_intrinsic else None,
                        c2w=c2w if load_c2w else None,
                        gt_bboxes=frame['gt_bboxes'],
                        dynamic_mask_seman=dynamic_mask_seman,
                        vehicle_points = frame["vehicle_points"],
                        vehicle_colors = frame["vehicle_colors"],
                         ))
            
    return cam_infos


def readWaymoInfo(path, white_background, eval, extension=".png", use_bg_gs=False, 
                  load_sky_mask = False, load_panoptic_mask = True, load_sam_mask = False,load_dynamic_mask = False,
                  load_feat_map = False,
                  load_intrinsic = False, load_c2w = False,
                  start_time = 0, end_time = -1, num_pts = 5000, 
                  save_occ_grid = False, occ_voxel_size = 0.4, recompute_occ_grid=True,
                  stride = 10 , original_start_time = 0, split_dynamic=True, args=None
                  ):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    load_size = [640, 960]
    # modified from emer-nerf
    data_root = path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder))/5
    start_time = start_time
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1
    # reconstruct each clip separately (no merge)
    original_start_time = start_time
    camera_list = [1,0,2]
    truncated_min_range, truncated_max_range = -2, 80
    cam_frustum_range = [0.01, 80]
    # set img_list
    load_sky_mask = load_sky_mask
    load_panoptic_mask = load_panoptic_mask
    load_sam_mask = load_sam_mask
    load_dynamic_mask = load_dynamic_mask
    load_feat_map = load_feat_map
    load_lidar, load_depthmap = True, True


    online_load = True
    if args.load_cache:
        cache_path = os.path.join(data_root, f"cache_{start_time}_{end_time}{'_gt_bbox' if args.load_gt_bbox else ''}.pkl")
        if os.path.exists(cache_path) and not args.force_reload:
            # read from cache
            print("Reading from cache...")
            online_load = False
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            time_line = cache['time_line']
            cam_to_worlds = cache['cam_to_worlds']
            intrinsics = cache['intrinsics']
            img_filepaths = cache['img_filepaths']
            # semantic_masks = cache['semantic_masks']
            instance_masks = cache['instance_masks']
            thing_masks = cache['thing_masks']
            vehicle_init_pose_dict = cache['vehicle_init_pose_dict']
            vehicle_previous_pose_dict = cache['vehicle_previous_pose_dict']
            gt_bboxes_list = cache['gt_bboxes_list']
            pred_boxes_list = cache['pred_boxes_list']
            pcd = cache['pcd']
            bg_pcd = cache['bg_pcd']
            road_pcd = cache['road_pcd']
            sky_pcd = cache['sky_pcd']
            vehicle_pcd_dict = cache['vehicle_pcd_dict']
            static_vehicle_pcd = cache['static_vehicle_pcd']
            occ_grid = cache['occ_grid']
            if not args.load_gt_bbox:
                static_ids = cache['static_ids']
                dynamic_ids = cache['dynamic_ids']
            aabb = cache['aabb']
            timestamps = cache['timestamps']
            timestamp_mapper = cache['timestamp_mapper']
            depth_maps = cache['depth_maps']
            dynamic_mask_filepaths = cache['dynamic_mask_filepaths']
            dynamic_mask_seman_list = cache['dynamic_mask_seman_list']
            dynamic_pcd = cache['dynamic_pcd']
            vehicle_points_list = []
            vehicle_colors_list = []
            vehicle_pcd = None
            # ------------------
            # get split: train and test splits from timestamps
            # ------------------
            # mask
            if stride != 0 :
                train_mask = (timestamps % int(stride) != 0) | (timestamps == 0)
            else:
                train_mask = np.ones(len(timestamps), dtype=bool)
            test_mask = ~train_mask
            # mask to index                                                                    
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            full_idx = np.arange(len(timestamps))
            train_timestamps = timestamps[train_mask]
            test_timestamps = timestamps[test_mask]
            

    if online_load:
        img_filepaths = []
        dynamic_mask_filepaths, sky_mask_filepaths = [], []
        semantic_mask_filepaths, instance_mask_filepaths = [], []
        semantic_masks, instance_masks = [], []
        thing_masks = []
        sam_mask_filepaths = []
        feat_map_filepaths = []
        dynamic_mask_filepaths = []
        lidar_filepaths = []
        vehicle_points_dict = {}
        vehicle_colors_dict = {}
        vehicle_init_pose_dict = {}
        vehicle_previous_pose_dict = {}
        gt_bboxes_list = []
        dynamic_mask_seman_list = []
        for t in range(start_time, end_time):
            for cam_idx in camera_list:
                img_filepaths.append(os.path.join(data_root, "images", f"{t:03d}_{cam_idx}.jpg"))
                dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
                sky_mask_filepaths.append(os.path.join(data_root, "sky_masks", f"{t:03d}_{cam_idx}.png"))
                if os.path.exists(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg")):
                    sam_mask_filepaths.append(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg"))
                if os.path.exists(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png")):
                    dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
                if load_feat_map:
                    feat_map_filepaths.append(os.path.join(data_root, "dinov2_vitb14", f"{t:03d}_{cam_idx}.npy"))
                    
            lidar_filepaths.append(os.path.join(data_root, "lidar", f"{t:03d}.bin"))

        img_filepaths = np.array(img_filepaths)
        dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        sky_mask_filepaths = np.array(sky_mask_filepaths)
        lidar_filepaths = np.array(lidar_filepaths)
        semantic_mask_filepaths = np.array(semantic_mask_filepaths)
        instance_mask_filepaths = np.array(instance_mask_filepaths)
        sam_mask_filepaths = np.array(sam_mask_filepaths)
        feat_map_filepaths = np.array(feat_map_filepaths)
        dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        # ------------------
        # construct timestamps
        # ------------------
        # original_start_time = 0
        idx_list = range(original_start_time, end_time)
        # map time to [0,1]
        timestamp_mapper = {}
        time_line = [i for i in idx_list]
        time_length = end_time - original_start_time - 1
        for index, time in enumerate(time_line):
            timestamp_mapper[time] = (time-original_start_time)/time_length
        max_time = max(timestamp_mapper.values())
        # ------------------
        # load poses: intrinsic, c2w, l2w
        # ------------------
        _intrinsics = []
        cam_to_egos = []
        for i in range(len(camera_list)):
            # load intrinsics
            intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # scale intrinsics w.r.t. load size
            fx, fy = (
                fx * load_size[1] / ORIGINAL_SIZE[i][1],
                fy * load_size[0] / ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx * load_size[1] / ORIGINAL_SIZE[i][1],
                cy * load_size[0] / ORIGINAL_SIZE[i][0],
            )
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            _intrinsics.append(intrinsic)
            # load extrinsics
            cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))
            # opencv coordinate system: x right, y down, z front
            # waymo coordinate system: x front, y left, z up
            cam_to_egos.append(cam_to_ego @ OPENCV2DATASET) # opencv_cam -> waymo_cam -> waymo_ego
        
        # LOAD TRACKING RESULT
        tracking_results = {}
        tracking_dir = os.path.join(data_root, 'tracking')
        for cam_id in camera_list:
            # TODO DELETE THE LAST MATRIX HERE, CLEAN TRACKING FILE IN FUTURE
            tracking_results[cam_id] = np.load(os.path.join(tracking_dir, f'view-{cam_id}.pkl'), allow_pickle=True)[:-1]
            
        
        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, cam_ids = [], []
        lidar_to_worlds = []
        # ===! for waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []
        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(os.path.join(data_root, "ego_pose", f"{start_time:03d}.txt"))
        for t in range(start_time, end_time):
            ego_to_world_current = np.loadtxt(os.path.join(data_root, "ego_pose", f"{t:03d}.txt"))
            # ego to world transformation: cur_ego -> world -> start_ego(world)
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in camera_list:
                cam_ids.append(cam_id)
                # transformation:
                # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[cam_id])
                # ===! we use time indices as the timestamp for waymo dataset for simplicity
                # ===! we can use the actual timestamps if needed
                # to be improved
                timestamps.append(t - start_time)
                timesteps.append(t - start_time)
            # lidar to world : lidar = ego in waymo
            lidar_to_worlds.append(ego_to_world)
        # convert to numpy arrays
        intrinsics = np.stack(intrinsics, axis=0)
        cam_to_worlds = np.stack(cam_to_worlds, axis=0)
        ego_to_worlds = np.stack(ego_to_worlds, axis=0)
        lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)
        cam_ids = np.array(cam_ids)
        timestamps = np.array(timestamps)
        timesteps = np.array(timesteps)
        # ------------------
        # get aabb: c2w --> frunstums --> aabb
        # ------------------
        # compute frustums
        frustums = []
        pix_corners = np.array( # load_size : [h, w]
            [[0,0],[0,load_size[0]],[load_size[1],load_size[0]],[load_size[1],0]]
        )
        for c2w, intri in zip(cam_to_worlds, intrinsics):
            frustum = []
            for cam_extent in cam_frustum_range:
                # pix_corners to cam_corners
                cam_corners = np.linalg.inv(intri) @ np.concatenate(
                    [pix_corners, np.ones((4, 1))], axis=-1
                ).T * cam_extent
                # cam_corners to world_corners
                world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
                # compute frustum
                frustum.append(world_corners)
            frustum = np.stack(frustum, axis=0)
            frustums.append(frustum)
        frustums = np.stack(frustums, axis=0)
        # compute aabb
        aabbs = []
        for frustum in frustums:
            flatten_frustum = frustum.transpose(0,2,1).reshape(-1,3)
            aabb_min = np.min(flatten_frustum, axis=0)
            aabb_max = np.max(flatten_frustum, axis=0)
            aabb = np.stack([aabb_min, aabb_max], axis=0)
            aabbs.append(aabb)
        aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
        aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
        print('cam frustum aabb min: ', aabb[0])
        print('cam frustum aabb max: ', aabb[1])
        # ------------------
        # get split: train and test splits from timestamps
        # ------------------
        # mask
        if stride != 0 :
            train_mask = (timestamps % int(stride) != 0) | (timestamps == 0)
        else:
            train_mask = np.ones(len(timestamps), dtype=bool)
        test_mask = ~train_mask
        # mask to index                                                                    
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        full_idx = np.arange(len(timestamps))
        train_timestamps = timestamps[train_mask]
        test_timestamps = timestamps[test_mask]
        # ------------------
        # load points and depth map
        # ------------------
        pts_path = os.path.join(data_root, "lidar")
        depth_maps = None
        # bg-gs settings
        #use_bg_gs = False
        bg_scale = 2.0 # used to scale fg-aabb
        if not os.path.exists(pts_path):
            print('no initial point cloud provided!')
            # random sample
            # Since this data set has no colmap data, we start with random points
            #num_pts = 2000
            print(f"Generating random point cloud ({num_pts})...")
            aabb_center = (aabb[0] + aabb[1]) / 2
            aabb_size = aabb[1] - aabb[0]
            # We create random points inside the bounds of the synthetic Blender scenes
            random_xyz = np.random.random((num_pts, 3)) 
            print('normed xyz min: ', np.min(random_xyz, axis=0))
            print('normed xyz max: ', np.max(random_xyz, axis=0))
            xyz = random_xyz * aabb_size + aabb[0]
            print('xyz min: ', np.min(xyz, axis=0))
            print('xyz max: ', np.max(xyz, axis=0))
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        else:
            # load lidar points
            # origins, directions, points, ranges, laser_ids = [], [], [], [], []
            points, dynamic_points, dynamic_point_masks = [], [], []
            thing_point_maps = []
            sky_points, vehicle_colors_list, vehicle_points_list = [], [], []
            vehicle_id_transform_dict = {}
            new_id = 0
            shs, dynamic_shs = [], []
            depth_maps = []
            accumulated_num_original_rays = 0
            accumulated_num_rays = 0
            pred_boxes_list = []
            for t in trange(0, len(lidar_filepaths), desc="loading lidar", dynamic_ncols=True):
                lidar_info = np.memmap(
                    lidar_filepaths[t],
                    dtype=np.float32,
                    mode="r",
                ).reshape(-1, 14)
                original_length = len(lidar_info)
                accumulated_num_original_rays += original_length
                lidar_origins = lidar_info[:, :3]
                lidar_points = lidar_info[:, 3:6]
                lidar_ids = lidar_info[:, -1]
                # select lidar points based on a truncated ego-forward-directional range
                # make sure most of lidar points are within the range of the camera
                valid_mask = lidar_points[:, 0] < truncated_max_range
                valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
                lidar_origins = lidar_origins[valid_mask]
                lidar_points = lidar_points[valid_mask]
                lidar_ids = lidar_ids[valid_mask]
                # transform lidar points to world coordinate system
                lidar_origins = (
                    lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                    + lidar_to_worlds[t][:3, 3:4]
                ).T
                lidar_points = (
                    lidar_to_worlds[t][:3, :3] @ lidar_points.T
                    + lidar_to_worlds[t][:3, 3:4]
                ).T
                pixel_point_mask_t = {}
                image_points_t = {}
                if load_depthmap:
                    # transform world-lidar to pixel-depth-map
                    rgb_point = np.random.random((len(lidar_points), 3))
                    gt_bboxes = []
                    if args.load_gt_bbox:
                        gt_bbox_path = os.path.join(data_root, "gt_bbox", f"{t:03d}.json")
                        with open(gt_bbox_path, 'r') as file:
                            gt_bbox_origin = json.load(file)
                        for box in gt_bbox_origin:
                            if box['type'] != 'vehicle' or box['is_moving'] == False:
                                continue
                            for key in ['translation', 'rotation', 'size']:
                                box[key] = np.array(box[key])
                            trs = np.linalg.inv(ego_to_world_start) @ \
                                np.concatenate([box['translation'], [1]])
                            box['translation'] = trs[:3]
                            box['rotation'] = np.linalg.inv(ego_to_world_start[:3, :3]) @ matrix_from_quaternion(box['rotation'])
                            gt_bboxes.append(box)
                    for idx, cam_idx in enumerate(camera_list):
                        if split_dynamic:
                            # dynamic_mask = Image.open(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
                            # dynamic_mask = dynamic_mask.resize(load_size[::-1], Image.NEAREST)
                            # dynamic_mask = np.array(dynamic_mask) / 255
                            semantic_mask_pil = Image.open(os.path.join(data_root, "panoptic", "semantic_mask", f"{t:03d}_{cam_idx}.png"))
                            semantic_mask_pil = semantic_mask_pil.resize(load_size[::-1], Image.NEAREST)
                            semantic_mask = np.array(semantic_mask_pil)
                            instance_mask_pil = Image.open(os.path.join(data_root, "panoptic", "instance_mask", f"{t:03d}_{cam_idx}.png"))
                            instance_mask_pil = instance_mask_pil.resize(load_size[::-1], Image.NEAREST)
                            instance_mask = np.array(instance_mask_pil)
                            semantic_masks.append(semantic_mask_pil)
                            instance_masks.append(instance_mask_pil)
                            dynamic_mask = np.zeros(load_size)
                            thing_mask = np.zeros(load_size)
                            car_id_mask = np.zeros(load_size)
                            thing_mask[:] = -1
                            for obj_id in STATIC_OBJECT_ID:
                                thing_mask[semantic_mask == obj_id] = THING.STATIC_OBJECT
                            for obj_id in DYNAMIC_OBJECT_ID:
                                thing_mask[semantic_mask == obj_id] = THING.DYNAMIC_OBJECT
                                dynamic_mask[semantic_mask == obj_id] = 1
                            for obj_id in VEHICLE_ID:
                                thing_mask[semantic_mask == obj_id] = THING.VEHICLE
                            for obj_id in FLAT_ID:
                                thing_mask[semantic_mask == obj_id] = THING.ROAD
                            thing_mask[semantic_mask == SEG_NAME2ID['Sky']] = THING.SKY
                        else:
                            thing_mask = np.zeros(load_size)
                            dynamic_mask = np.zeros(load_size)
                        
                        thing_masks.append(thing_mask)
                        
                        image = Image.open(os.path.join(data_root, "images", f"{t:03d}_{cam_idx}.jpg"))
                        image = image.resize(load_size[::-1], Image.NEAREST)
                        image = np.array(image) / 255
                        dynamic_point_mask = np.zeros([lidar_points.shape[0]])
                        thing_point_map = np.zeros([lidar_points.shape[0]])
                        vis_mask = np.zeros([lidar_points.shape[0]])
                        
                        # world-lidar-pts --> camera-pts : w2c
                        c2w = cam_to_worlds[int(len(camera_list))*t + idx]
                        w2c = np.linalg.inv(c2w)
                        cam_points = (
                            w2c[:3, :3] @ lidar_points.T
                            + w2c[:3, 3:4]
                        ).T
                        # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
                        pixel_points = (
                            intrinsics[int(len(camera_list))*t + idx] @ cam_points.T
                        ).T
                        pixel_points_mask1 = pixel_points[:, 2]>0
                        # select points in front of the camera
                        pixel_points = pixel_points[pixel_points_mask1]
                        # normalize pixel points : (u,v,1)
                        image_points = pixel_points[:, :2] / pixel_points[:, 2:]
                        # filter out points outside the image
                        valid_mask = (
                            (image_points[:, 0] >= 0)
                            & (image_points[:, 0] < load_size[1])
                            & (image_points[:, 1] >= 0)
                            & (image_points[:, 1] < load_size[0])
                        )
                        pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
                        image_points = image_points[valid_mask]     # pts_img : (u,v)
                        # compute depth map
                        depth_map = np.zeros(load_size)
                        depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]
                        # sampling points with smallest distance, since points are sparse, discard this
                        # depth_sort_idx = np.argsort(pixel_points[:,2])[::-1]
                        # image_points = image_points[depth_sort_idx]
                        # pixel_points = pixel_points[depth_sort_idx]
                        # depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]
                        depth_maps.append(depth_map)
                        
                        pixel_points_mask1[pixel_points_mask1] = valid_mask

                        dynamic_point_mask[pixel_points_mask1] += \
                            dynamic_mask[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)]
                        
                        rgb_point[pixel_points_mask1, :] = image[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)]
                        vis_mask[pixel_points_mask1] = 1
                        thing_point_map[pixel_points_mask1] = thing_mask[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)]

                        
                        vehicle_moving_mask = np.zeros(len(pixel_points))
                        
                        if args.load_gt_bbox:
                            gt_bboxes_vis = []
                            for box in gt_bboxes:
                                box_center = box['translation']
                                camera_center = w2c[:3, :3] @ box_center + w2c[:3, 3:4].T
                                proj_center = intrinsics[int(len(camera_list))*t + idx] @ camera_center.T
                                proj_center = proj_center[:2] / proj_center[2]
                                proj_center = proj_center.squeeze()
                                valid = (
                                (proj_center[0] >= 0)
                                & (proj_center[0] < load_size[1])
                                & (proj_center[1] >= 0)
                                & (proj_center[1] < load_size[0])
                                )   
                                proj_center = proj_center.astype(np.int32)
                                if not valid:
                                    continue
                                if thing_mask[proj_center[1], proj_center[0]] != THING.VEHICLE:
                                    continue
                                gt_bboxes_vis.append(box)
                                instance_id = instance_mask[proj_center[1], proj_center[0]]
                                cid = semantic_mask[proj_center[1], proj_center[0]]
                                box_mask = (instance_mask == instance_id) & (semantic_mask == cid)
                                dynamic_mask[box_mask] = True
                                box_point_mask = box_mask[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)]
                                color = image[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)]
                                box_lidar_points = lidar_points[pixel_points_mask1][box_point_mask]
                                diameter = np.linalg.norm(box['size'])
                                thr = diameter / 2 * args.vehicle_extent
                                center_dist = np.linalg.norm(box_lidar_points.T - np.expand_dims(box['translation'], -1), axis=0)
                                box_point_mask[box_point_mask] = center_dist < thr
                                box_lidar_points = lidar_points[pixel_points_mask1][box_point_mask]
                                color_lidar_points = color[box_point_mask]

                                vehicle_moving_mask[box_point_mask] = 1
                                gid = box['gid']
                                if gid not in vehicle_points_dict.keys():
                                    vehicle_init_pose_dict[gid] = box
                                    vehicle_points_dict[gid] = box_lidar_points
                                    vehicle_colors_dict[gid] = color_lidar_points
                                else:
                                    init_box = vehicle_init_pose_dict[gid]
                                    canonicalized_lidar_points = box['rotation'].T @ (box_lidar_points.T - np.expand_dims(box['translation'], -1))
                                    init_lidar_points = init_box['rotation'] @ canonicalized_lidar_points + np.expand_dims(init_box['translation'], -1)
                                    vehicle_points_dict[gid] = np.concatenate([vehicle_points_dict[gid], init_lidar_points.T], axis=0)
                                    vehicle_colors_dict[gid] = np.concatenate([vehicle_colors_dict[gid], color_lidar_points], axis=0)
                        else:
                            pixel_point_mask_t[cam_idx] = pixel_points_mask1
                            image_points_t[cam_idx] = image_points
                        gt_bboxes_list.append(gt_bboxes)
                        vis_mask[pixel_points_mask1] = (vis_mask[pixel_points_mask1] > 0) & (vehicle_moving_mask == 0)
                        dynamic_mask_seman_list.append(dynamic_mask)
                    
                    def icp_registration(source, target, init_pose, max_correspondence_distance=0.2):
                        """使用ICP进行点云配准"""
                        source_pcd = o3d.geometry.PointCloud()
                        source_pcd.points = o3d.utility.Vector3dVector(source)
                        
                        target_pcd = o3d.geometry.PointCloud()
                        target_pcd.points = o3d.utility.Vector3dVector(target)

                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            source_pcd, target_pcd, max_correspondence_distance,
                            init=init_pose,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
                        
                        return reg_p2p.transformation
                    

                    def box_size_crop(pc, size, center=None):
                        center = np.median(pc, 0) if center is None else center
                        size = size / 2
                        distance = np.abs(pc - center)
                        out_mask = distance > size
                        out_mask = out_mask.sum(-1).astype(bool)
                        in_mask = ~out_mask
                        return in_mask

                    # ASSOCIATE TRACKING RESULT IN 3 FRAMES & INIT VEHICLE POINT CLOUD 
                    vehicle_pcd_save_dir = os.path.join(data_root, 'vehicle_pcd')
                    os.makedirs(vehicle_pcd_save_dir, exist_ok=True)
                    pred_boxes = {}
                    if not args.load_gt_bbox:
                        tracking_t = {}
                        curr_vehicle_id_list = []
                        for cam_id in camera_list:
                            curr_cam_dict = {}
                            image_points_i = image_points_t[cam_id]
                            pixel_point_mask_i = pixel_point_mask_t[cam_id]
                            for res in tracking_results[cam_id]:
                                if res['image_id'] == t + start_time:
                                    mask_res = np.array(mask_util.decode(res['mask'])).astype(float)
                                    res['mask'] = mask_res
                                    lidar_point_mask = pixel_point_mask_i.copy()
                                    lidar_point_mask[lidar_point_mask] = res['mask'][image_points_i[:, 1].astype(np.int32), image_points_i[:, 0].astype(np.int32)]
                                    res['lidar_point_mask'] = lidar_point_mask
                                    curr_cam_dict[res['track_id']] = res
                                    curr_vehicle_id_list.append((cam_id, res['track_id']))
                            tracking_t[cam_id] = curr_cam_dict

                        OVERLAP_POINT_THRESHOLD = args.overlap_point_threshold
                        for i_cam in range(len(camera_list)):
                            image_points_i = image_points_t[i_cam]
                            pixel_point_mask_i = pixel_point_mask_t[i_cam]
                            for obj1 in tracking_t[i_cam].values():
                                obj1_lidar_point_mask = obj1['lidar_point_mask']
                                obj1_id = obj1['track_id']
                                obj1_mask = obj1['mask']
                                is_associate = False
                                for j_cam in range(i_cam+1, len(camera_list)):
                                    for obj2 in tracking_t[j_cam].values():
                                        obj2_lidar_point_mask = obj2['lidar_point_mask']
                                        overlap_num = sum(obj1_lidar_point_mask & obj2_lidar_point_mask)
                                        if overlap_num > OVERLAP_POINT_THRESHOLD:
                                            is_associate = True
                                            obj2_id = obj2['track_id']
                                            obj2_mask = obj2['mask']
                                            if (i_cam, obj1_id) in vehicle_id_transform_dict and (j_cam, obj2_id) in vehicle_id_transform_dict:
                                                common_id = vehicle_id_transform_dict[(i_cam, obj1_id)]
                                                change_id = vehicle_id_transform_dict[(j_cam, obj2_id)]
                                                if change_id != common_id:
                                                    for k, v in vehicle_id_transform_dict.items():
                                                        if v == change_id:
                                                            vehicle_id_transform_dict[k] = common_id
                                            elif (i_cam, obj1_id) in vehicle_id_transform_dict:
                                                vehicle_id_transform_dict[(j_cam, obj2_id)] = vehicle_id_transform_dict[(i_cam, obj1_id)]
                                            elif (j_cam, obj2_id) in vehicle_id_transform_dict:
                                                vehicle_id_transform_dict[(i_cam, obj1_id)] = vehicle_id_transform_dict[(j_cam, obj2_id)]
                                            else:
                                                vehicle_id_transform_dict[(i_cam, obj1_id)] = new_id
                                                vehicle_id_transform_dict[(j_cam, obj2_id)] = new_id
                                                new_id += 1
                                            break
                                if (not is_associate) and ((i_cam, obj1_id) not in vehicle_id_transform_dict):
                                    vehicle_id_transform_dict[(i_cam, obj1_id)] = new_id
                                    new_id += 1
                        
                        VEHICLE_SIZE = {'car': np.array([8,4,4]), 
                                        'bus': np.array([40,10,10]), 
                                        'truck': np.array([40,10,10])}
                        # INITIALIZE VEHICLE PC
                        # new_id is total id number
                        MIN_POINT_THRESHOLD = args.vehicle_min_point_threshold
                        for vehicle_id in range(new_id):
                            lidar_point_mask = np.zeros([len(lidar_points)]).astype(bool)
                            for k, v in vehicle_id_transform_dict.items():
                                if v == vehicle_id:
                                    if k not in curr_vehicle_id_list:
                                        continue
                                    cam_id, obj_id = k
                                    obj = tracking_t[cam_id][obj_id]
                                    lidar_point_mask = lidar_point_mask | obj['lidar_point_mask']
                            if (not lidar_point_mask.any()) or lidar_point_mask.sum() < MIN_POINT_THRESHOLD:
                                continue
                            curr_vehicle_points = lidar_points[lidar_point_mask]
                            curr_vehicle_colors = rgb_point[lidar_point_mask]

                            if vehicle_id not in vehicle_points_dict:
                                size = VEHICLE_SIZE[obj['category_id']]
                                in_box_mask = box_size_crop(curr_vehicle_points, size)
                                if in_box_mask.sum() < MIN_POINT_THRESHOLD:
                                    continue
                                vehicle_points_dict[vehicle_id] = curr_vehicle_points[in_box_mask]
                                vehicle_colors_dict[vehicle_id] = curr_vehicle_colors[in_box_mask]
                                #  t = center, s = pre-defined
                                box = {'translation': np.mean(curr_vehicle_points, 0),
                                    'rotation': np.eye(3),
                                    'size': VEHICLE_SIZE[obj['category_id']]}
                                rel_box = {'translation': np.array([0,0,0]),
                                            'rotation': np.eye(3),
                                            'size': VEHICLE_SIZE[obj['category_id']]}
                                pred_boxes[vehicle_id] = rel_box
                                vehicle_init_pose_dict[vehicle_id] = box
                                vehicle_previous_pose_dict[vehicle_id] = rel_box
                                # storePly(os.path.join(vehicle_pcd_save_dir, f'{vehicle_id}_{t}.ply'), vehicle_points_dict[vehicle_id], vehicle_colors_dict[vehicle_id] * 255)

                            else:
                                canonical_model = vehicle_points_dict[vehicle_id]
                                prev_rot = vehicle_previous_pose_dict[vehicle_id]['rotation']
                                init_pose = np.eye(4)
                                init_pose[:3,:3] = prev_rot
                                init_pose[:3, 3] = (np.median(curr_vehicle_points, 0) - np.median(canonical_model, 0)).T
                                pose_t = icp_registration(canonical_model, curr_vehicle_points, init_pose, max_correspondence_distance=args.icp_corr_dist)
                                if np.isnan(pose_t).any():
                                    pose_t = init_pose
                                rel_box = \
                                    {'rotation': pose_t[:3, :3],
                                    'translation': pose_t[:3, 3],
                                    'size': vehicle_previous_pose_dict[vehicle_id]['size']}
                                vehicle_previous_pose_dict[vehicle_id] = rel_box
                                pred_boxes[vehicle_id] = rel_box
                                canonical_curr_points = np.dot(pose_t[:3, :3].T, (curr_vehicle_points - pose_t[:3, 3]).T).T
                                # canonical_curr_points = np.dot(pose_t[:3, :3], curr_vehicle_points.T).T + pose_t[:3, 3]
                                canonical_model_update = np.concatenate([canonical_model, canonical_curr_points])
                                in_box_mask = box_size_crop(canonical_model_update, vehicle_previous_pose_dict[vehicle_id]['size'])

                                if in_box_mask.sum() < MIN_POINT_THRESHOLD:
                                    vehicle_points_dict.pop(vehicle_id)
                                    vehicle_colors_dict.pop(vehicle_id)
                                    vehicle_previous_pose_dict.pop(vehicle_id)
                                    vehicle_init_pose_dict.pop(vehicle_id)
                                else:
                                    vehicle_points_dict[vehicle_id] = canonical_model_update[in_box_mask]
                                    vehicle_colors_dict[vehicle_id] = np.concatenate([vehicle_colors_dict[vehicle_id], curr_vehicle_colors])[in_box_mask]
                                    # storePly(os.path.join(vehicle_pcd_save_dir, f'{vehicle_id}_{t}.ply'), vehicle_points_dict[vehicle_id], vehicle_colors_dict[vehicle_id] * 255)

                    pred_boxes_list.append(pred_boxes)

                # # compute lidar directions
                # lidar_directions = lidar_points - lidar_origins
                # lidar_ranges = np.linalg.norm(lidar_directions, axis=-1, keepdims=True)
                # lidar_directions = lidar_directions / lidar_ranges
                # # time indices as timestamp
                # #lidar_timestamps = np.ones_like(lidar_ranges).squeeze(-1) * t
                # accumulated_num_rays += len(lidar_ranges)

                # origins.append(lidar_origins)
                # directions.append(lidar_directions)
                # ranges.append(lidar_ranges)
                # laser_ids.append(lidar_ids)
                
                # points.append(lidar_points)
                sh = RGB2SH(rgb_point)
                if args.filter_vis_point:
                    points.append(lidar_points[(vis_mask > 0)])
                    shs.append(sh[(vis_mask > 0)])
                    thing_point_maps.append(thing_point_map[(vis_mask > 0)])
                else:
                    points.append(lidar_points)
                    shs.append(sh)
                    thing_point_maps.append(thing_point_map)
                


            #origins = np.concatenate(origins, axis=0)
            #directions = np.concatenate(directions, axis=0)
            #ranges = np.concatenate(ranges, axis=0)
            #laser_ids = np.concatenate(laser_ids, axis=0)
            points = np.concatenate(points, axis=0)
            shs = np.concatenate(shs, axis=0)
            thing_point_maps = np.concatenate(thing_point_maps, axis=0)
            # filter points by cam_aabb 
            cam_aabb_mask = np.all((points >= aabb[0]) & (points <= aabb[1]), axis=-1)
            points = points[cam_aabb_mask]
            shs = shs[cam_aabb_mask]
            thing_point_maps = thing_point_maps[cam_aabb_mask]
            points_all = points.copy()
            shs_all = shs.copy()
            # static point
            static_point_maps = (thing_point_maps == THING.STATIC_OBJECT) | (thing_point_maps == THING.ROAD) | (thing_point_maps == THING.VEHICLE)
            points = points_all[static_point_maps]
            shs = shs_all[static_point_maps]
            static_thing_maps = thing_point_maps[static_point_maps]
            # dynamic point
            dynamic_point_maps = (thing_point_maps == THING.DYNAMIC_OBJECT) 
            dynamic_points = points_all[dynamic_point_maps]
            dynamic_shs = shs_all[dynamic_point_maps]
            dynamic_thing_maps = thing_point_maps[dynamic_point_maps]

            # construct occupancy grid to aid densification
            if save_occ_grid:
                #occ_grid_shape = (int(np.ceil((aabb[1, 0] - aabb[0, 0]) / occ_voxel_size)),
                #                    int(np.ceil((aabb[1, 1] - aabb[0, 1]) / occ_voxel_size)),
                #                    int(np.ceil((aabb[1, 2] - aabb[0, 2]) / occ_voxel_size)))
                if not os.path.exists(os.path.join(data_root, "occ_grid.npy")) or recompute_occ_grid:
                    occ_grid = get_OccGrid(points, aabb, occ_voxel_size)
                    np.save(os.path.join(data_root, "occ_grid.npy"), occ_grid)
                else:
                    occ_grid = np.load(os.path.join(data_root, "occ_grid.npy"))
                print(f'Lidar points num : {len(points)}')
                print("occ_grid shape : ", occ_grid.shape)
                print(f'occ voxel num :{occ_grid.sum()} from {occ_grid.size} of ratio {occ_grid.sum()/occ_grid.size}')
            
            # downsample points
            # points,shs = GridSample3D(points, shs)
            # print("grid sampled points: ", points.shape)

            print(f'original downsampled point {len(points)}')
            if len(points)>num_pts:
                downsampled_indices = np.random.choice(
                    len(points), num_pts, replace=False
                )
                points = points[downsampled_indices]
                shs = shs[downsampled_indices]
                static_thing_maps = static_thing_maps[downsampled_indices]
                
            # check
            #voxel_coords = np.floor((points - aabb[0]) / occ_voxel_size).astype(int)
            #occ = occ_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
            #origins = origins[downsampled_indices] 
            
            ## 计算 points xyz 的范围
            xyz_min = np.min(points,axis=0)
            xyz_max = np.max(points,axis=0)
            print("init lidar xyz min:",xyz_min)
            print("init lidar xyz max:",xyz_max)        # lidar-points aabb (range)
            ## 设置 背景高斯点
            if use_bg_gs:
                fg_aabb_center, fg_aabb_size = (aabb[0] + aabb[1]) / 2, aabb[1] - aabb[0] # cam-frustum aabb
                # use bg_scale to scale the aabb
                bg_gs_aabb = np.stack([fg_aabb_center - fg_aabb_size * bg_scale / 2, 
                            fg_aabb_center + fg_aabb_size * bg_scale / 2], axis=0)
                bg_aabb_center, bg_aabb_size = (bg_gs_aabb[0] + bg_gs_aabb[1]) / 2, bg_gs_aabb[1] - bg_gs_aabb[0]
                # add bg_gs_aabb SURFACE points
                bg_points = sample_on_aabb_surface(bg_aabb_center, bg_aabb_size, 1000)
                print("bg_gs_points min:",np.min(bg_points,axis=0))
                print("bg_gs_points max:",np.max(bg_points,axis=0))
                # DO NOT add bg_gs_points to points
                #points = np.concatenate([points, bg_points], axis=0)
                #shs = np.concatenate([shs, np.random.random((len(bg_points), 3)) / 255.0], axis=0)
                bg_shs = np.random.random((len(bg_points), 3)) / 255.0
                # visualize
                #from utils.general_utils import visualize_points
                #visualize_points(points, fg_aabb_center, fg_aabb_size)
                
            if split_dynamic:
                road_pcd = BasicPointCloud(points=points[static_thing_maps==THING.ROAD], colors=SH2RGB(shs[static_thing_maps==THING.ROAD]), 
                                        normals=np.zeros((len(points[static_thing_maps==THING.ROAD]), 3)))
                z_max = xyz_max[2]
                sky_pts_x_num = int(np.sqrt(args.sky_pt_num))
                sky_pts_num = sky_pts_x_num ** 2
                sky_points = np.zeros([sky_pts_num, 3])
                sky_colors = np.ones([sky_pts_num, 3])
                x_sky = np.linspace(aabb[0,0], aabb[1,0], sky_pts_x_num)
                y_sky = np.linspace(aabb[0,1], aabb[1,1], sky_pts_x_num)
                xv, yv = np.meshgrid(x_sky, y_sky)
                sky_points[:, 0] = xv.flatten()
                sky_points[:, 1] = yv.flatten()
                sky_points[:, 2] = args.sky_height
                sky_pcd = BasicPointCloud(points=sky_points, colors=sky_colors, 
                                        normals=np.zeros((sky_pts_num, 3)))
                dynamic_pcd = BasicPointCloud(points=dynamic_points, colors=SH2RGB(dynamic_shs), normals=np.zeros((len(dynamic_points), 3))) 

                if args.load_gt_bbox:
                    static_vehicle_points = points[static_thing_maps==THING.VEHICLE]
                    static_vehicle_shs = shs[static_thing_maps==THING.VEHICLE]
                    static_vehicle_pcd = BasicPointCloud(points=static_vehicle_points, colors=SH2RGB(static_vehicle_shs), 
                                            normals=np.zeros_like(static_vehicle_points))
                    vehicle_pcd = None
                    vehicle_pcd_dict = {}
                    for key in vehicle_points_dict.keys():
                        vehicle_pcd_dict[key] = BasicPointCloud(points=vehicle_points_dict[key], colors=vehicle_colors_dict[key], 
                                            normals=np.zeros_like(vehicle_points_dict[key]))
                    points = points[static_thing_maps==THING.STATIC_OBJECT]
                    shs = shs[static_thing_maps==THING.STATIC_OBJECT]
                else:
                    points = points[static_thing_maps==THING.STATIC_OBJECT]
                    shs = shs[static_thing_maps==THING.STATIC_OBJECT]
                    # try to know which vehicle is moving
                    MOVING_THRESHOLD = args.vehicle_moving_threshold
                    vehicle_pcd_dict = {}
                    static_vehicle_points = []
                    static_vehicle_colors = []
                    static_ids = []
                    dynamic_ids = []
                    
                    for vehicle_id in range(new_id):
                        if vehicle_id not in vehicle_points_dict:
                            continue
                        if np.linalg.norm(vehicle_previous_pose_dict[vehicle_id]['translation']) < MOVING_THRESHOLD:
                            # static
                            static_vehicle_points.append(vehicle_points_dict[vehicle_id])
                            static_vehicle_colors.append(vehicle_colors_dict[vehicle_id])
                            vehicle_init_pose_dict.pop(vehicle_id)
                            static_ids.append(vehicle_id)
                        else:
                            # dynamic
                            vehicle_pcd_dict[vehicle_id] = BasicPointCloud(points=vehicle_points_dict[vehicle_id], colors=vehicle_colors_dict[vehicle_id], 
                                normals=np.zeros_like(vehicle_points_dict[vehicle_id]))
                            storePly(os.path.join(vehicle_pcd_save_dir, f'{vehicle_id}.ply'), vehicle_points_dict[vehicle_id], vehicle_colors_dict[vehicle_id] * 255)
                            dynamic_ids.append(vehicle_id)
                    # modify dynamic mask by setting all static vehicle points to 0
                    dynamic_track_obj_ids = {k:[] for k in camera_list}
                    for vehicle_id in dynamic_ids:
                        for k, v in vehicle_id_transform_dict.items():
                            if v == vehicle_id:
                                cam_idx, obj_id = k
                                dynamic_track_obj_ids[cam_idx].append(obj_id)
                    for t in range(end_time-start_time):
                        for idx, cam_idx in enumerate(camera_list):
                            img_idx = int(len(camera_list))*t + idx
                            dynamic_mask = dynamic_mask_seman_list[img_idx]
                            for obj_id in dynamic_track_obj_ids[cam_idx]:
                                for res in tracking_results[cam_idx]:
                                    if res['track_id'] == obj_id and res['image_id'] == t + start_time:
                                        mask = res['mask']
                                        dynamic_mask[mask.astype(bool)] = 1
                            dynamic_mask_seman_list[img_idx] = dynamic_mask

                    static_vehicle_points = np.concatenate(static_vehicle_points)
                    static_vehicle_colors = np.concatenate(static_vehicle_colors)
                    static_vehicle_pcd = BasicPointCloud(points=static_vehicle_points, colors=static_vehicle_colors, 
                                            normals=np.zeros_like(static_vehicle_points))
                    storePly(os.path.join(vehicle_pcd_save_dir, f'static.ply'), static_vehicle_points, static_vehicle_colors * 255)
                    vehicle_pcd = None
                    gt_bboxes_list = []
                    for boxes in pred_boxes_list:
                        gt_bboxes = []
                        for k, v in boxes.items():
                            v['gid'] = k
                            gt_bboxes.append(v)
                        for _ in range(len(camera_list)):
                            gt_bboxes_list.append(gt_bboxes)
                    
            else:
                dynamic_pcd = None
                road_pcd = None
                sky_pcd = None
                vehicle_pcd_dict = None
                
            # save ply
            ply_path = os.path.join(data_root, "ds-points3d.ply")
            storePly(ply_path, points, SH2RGB(shs) * 255)
            pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=np.zeros((len(points), 3)))  
            if use_bg_gs:
                bg_ply_path = os.path.join(data_root, "ds-bg-points3d.ply")
                storePly(bg_ply_path, bg_points, SH2RGB(bg_shs) * 255)
                bg_pcd = BasicPointCloud(points=bg_points, colors=SH2RGB(bg_shs), normals=np.zeros((len(bg_points), 3)))
            else:
                bg_pcd, bg_ply_path = None, None
            # load depth maps
            if load_depthmap:
                assert depth_maps is not None, "should not use random-init-gs, ans set load_depthmap=True"
                depth_maps = np.stack(depth_maps, axis=0)
            
       
    if online_load and args.load_cache:
        # save cache
        print("saving cache at {}".format(cache_path))
        cache = {}
        cache['time_line'] = time_line
        cache['cam_to_worlds'] = cam_to_worlds
        cache['img_filepaths'] = img_filepaths
        cache['intrinsics'] = intrinsics
        cache['depth_maps'] = depth_maps
        cache['thing_masks'] = thing_masks
        cache['instance_masks'] = instance_masks
        cache['dynamic_mask_seman_list'] = dynamic_mask_seman_list
        # cache['vehicle_points_dict'] = vehicle_points_dict
        # cache['vehicle_colors_dict'] = vehicle_colors_dict
        cache['vehicle_init_pose_dict'] = vehicle_init_pose_dict 
        cache['vehicle_previous_pose_dict'] = vehicle_previous_pose_dict
        cache['pcd'] = pcd
        cache['bg_pcd'] = bg_pcd
        cache['road_pcd'] = road_pcd
        cache['sky_pcd'] = sky_pcd
        cache['dynamic_pcd'] = dynamic_pcd
        cache['vehicle_pcd_dict'] = vehicle_pcd_dict
        cache['static_vehicle_pcd'] = static_vehicle_pcd
        cache['gt_bboxes_list'] = gt_bboxes_list
        cache['pred_boxes_list'] = pred_boxes_list
        if not args.load_gt_bbox:
            cache['static_ids'] = static_ids
            cache['dynamic_ids'] = dynamic_ids
        cache['occ_grid'] = occ_grid
        cache['aabb'] = aabb
        cache['timestamp_mapper'] = timestamp_mapper
        cache['timestamps'] = timestamps
        cache['dynamic_mask_filepaths'] = dynamic_mask_filepaths
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)


    # ------------------
    # prepare cam-pose dict
    # ------------------
    train_frames_list = [] # time, transform_matrix(c2w), img_path
    test_frames_list = []
    full_frames_list = []
    for idx, t in enumerate(train_timestamps):
        frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引
                            transform_matrix = cam_to_worlds[train_idx[idx]],
                            file_path = img_filepaths[train_idx[idx]],
                            intrinsic = intrinsics[train_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[train_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[train_idx[idx]] if load_depthmap else None,
                            semantic_mask = thing_masks[train_idx[idx]] if load_panoptic_mask else None,
                            instance_mask = instance_masks[train_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[train_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[train_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[train_idx[idx]] if load_dynamic_mask else None,
                            gt_bboxes = gt_bboxes_list[train_idx[idx]] if len(gt_bboxes_list)>0 else None,
                            dynamic_mask = dynamic_mask_seman_list[train_idx[idx]] if len(dynamic_mask_seman_list)>0 else None,
                            vehicle_points = vehicle_points_list[train_idx[idx]] if len(vehicle_points_list)>0 else None,
                            vehicle_colors=vehicle_colors_list[train_idx[idx]] if len(vehicle_colors_list)>0 else None,
        )
        train_frames_list.append(frame_dict)
    for idx, t in enumerate(test_timestamps):
        frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引 
                            transform_matrix = cam_to_worlds[test_idx[idx]],
                            file_path = img_filepaths[test_idx[idx]],
                            intrinsic = intrinsics[test_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[test_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[test_idx[idx]] if load_depthmap else None,
                            semantic_mask = thing_masks[test_idx[idx]] if load_panoptic_mask else None,
                            instance_mask = instance_masks[test_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[test_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[test_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[test_idx[idx]] if load_dynamic_mask else None,
                            gt_bboxes = gt_bboxes_list[test_idx[idx]] if len(gt_bboxes_list)>0 else None,
                            dynamic_mask = dynamic_mask_seman_list[test_idx[idx]] if len(dynamic_mask_seman_list)>0 else None,
                            vehicle_points = vehicle_points_list[test_idx[idx]] if len(vehicle_points_list)>0 else None,
                            vehicle_colors=vehicle_colors_list[test_idx[idx]] if len(vehicle_colors_list)>0 else None,
        )
        test_frames_list.append(frame_dict)
    if len(test_timestamps)==0:
        full_frames_list = train_frames_list
    else:
        for idx, t in enumerate(timestamps):
            frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引 
                                transform_matrix = cam_to_worlds[full_idx[idx]],
                                file_path = img_filepaths[full_idx[idx]],
                                intrinsic = intrinsics[full_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[full_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[full_idx[idx]] if load_depthmap else None,
                                semantic_mask = thing_masks[full_idx[idx]] if load_panoptic_mask else None,
                                instance_mask = instance_masks[full_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[full_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[full_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[full_idx[idx]] if load_dynamic_mask else None,
                                gt_bboxes = gt_bboxes_list[full_idx[idx]] if len(gt_bboxes_list)>0 else None,
                                dynamic_mask = dynamic_mask_seman_list[full_idx[idx]] if len(dynamic_mask_seman_list)>0 else None,
                                vehicle_points = vehicle_points_list[full_idx[idx]] if len(vehicle_points_list)>0 else None,
                                vehicle_colors=vehicle_colors_list[full_idx[idx]] if len(vehicle_colors_list)>0 else None,
            )
            full_frames_list.append(frame_dict)
    
    # ------------------
    # load cam infos: image, c2w, intrinsic, load_size
    # ------------------
    print("Reading Training Transforms")
    train_cam_infos = constructCameras_waymo(train_frames_list, white_background, timestamp_mapper, 
                                             load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,
                                             original_start_time=original_start_time)
    print("Reading Test Transforms")
    test_cam_infos = constructCameras_waymo(test_frames_list, white_background, timestamp_mapper,
                                            load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,
                                            original_start_time=original_start_time)
    print("Reading Full Transforms")
    # full_cam_infos = constructCameras_waymo(full_frames_list, white_background, timestamp_mapper,
    #                                         load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    full_cam_infos = copy.copy(train_cam_infos)
    full_cam_infos.extend(test_cam_infos)
    
    #print("Generating Video Transforms")
    #video_cam_infos = generateCamerasFromTransforms_waymo(test_frames_list, max_time)
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)


    #     for cam in train_cam_infos+test_cam_infos:
    #         if cam.semantic_mask is not None and cam.instance_mask is not None:
    #             panoptic_object_ids = get_panoptic_id(cam.semantic_mask, cam.instance_mask).unique()
    #             panoptic_object_ids_list.append(panoptic_object_ids)
    #     # get unique panoptic_objects_ids
    #     panoptic_object_ids = torch.cat(panoptic_object_ids_list).unique().sort()[0].tolist()
    #     num_panoptic_objects = len(panoptic_object_ids)
    #     # map panoptic_id to idx
    #     for idx, panoptic_id in enumerate(panoptic_object_ids):
    #         panoptic_id_to_idx[panoptic_id] = idx

    
    scene_info = SceneInfo(point_cloud=pcd,
                           bg_point_cloud=bg_pcd,
                           dynamic_point_cloud=dynamic_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           full_cameras=full_cam_infos,
                           #video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           # background settings
                        #    ply_path=pts_path,
                        #    bg_ply_path=bg_ply_path,
                           cam_frustum_aabb=aabb,
                           # panoptic segs
                        #    num_panoptic_objects=num_panoptic_objects,
                        #    panoptic_object_ids=panoptic_object_ids,
                        #    panoptic_id_to_idx=panoptic_id_to_idx,
                           # occ grid
                           occ_grid=occ_grid if save_occ_grid else None,
                           road_pcd = road_pcd,
                           sky_pcd = sky_pcd,
                           static_vehicle_pcd=static_vehicle_pcd,
                           vehicle_pcd_dict=vehicle_pcd_dict,
                           vehicle_init_pose_dict=vehicle_init_pose_dict,
                           vehicle_pcd=vehicle_pcd
                           )

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo" : readWaymoInfo,
}

def GridSample3D(in_pc,in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)
    
    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:,1] * pc_boundary[2] + quantized_pc[:,2]
    
    split_point, index = get_split_point(voxel_index)
    
    in_points = in_pc[index,:]
    out_points = in_points[split_point[:-1],:]
    
    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]
    
    # 创建一个新的BasicPointCloud实例作为输出
    # out_pc =out_points
    # #remap index in_pc to out_pc
    # remap = np.zeros(in_pc.points.shape[0])
        
    # for ind in range(len(split_point)-1):
    #     cur_start = split_point[ind]
    #     cur_end = split_point[ind+1]
    #     remap[cur_start:cur_end] = ind
    
    # remap_back = remap.copy()
    # remap_back[index] = remap
    
    # remap_back = remap_back.astype(np.int64)
    return out_points,out_colors

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index
