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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.nn import functional as F
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.full_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            #scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"frame_info.json")):
            print("Found frame_info.json file, assuming Waymo data set!")
            scene_info = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.white_background, args.eval,
                                    use_bg_gs = False,
                                    load_sky_mask = args.load_sky_mask, #False, 
                                    load_panoptic_mask = args.load_panoptic_mask, #True, 
                                    load_intrinsic = args.load_intrinsic, #False,
                                    load_c2w = args.load_c2w, #False,
                                    load_sam_mask = args.load_sam_mask, #False,
                                    load_dynamic_mask = args.load_dynamic_mask, #False,
                                    load_feat_map = args.load_feat_map, #False,
                                    start_time = args.start_time, #0,
                                    end_time = args.end_time, # 100,
                                    num_pts = args.num_pts,
                                    save_occ_grid = args.save_occ_grid,
                                    occ_voxel_size = args.occ_voxel_size,
                                    recompute_occ_grid = args.recompute_occ_grid,
                                    stride = args.stride,
                                    original_start_time = args.original_start_time,
                                    split_dynamic = args.split_dynamic,
                                    args = args
                                    )
            dataset_type="waymo"
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            #with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = max(scene_info.nerf_normalization["radius"], 10)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Full Cameras")
            self.full_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.full_cameras, resolution_scale, args)

        if self.loaded_iter:
            raise NotImplementedError("Loading of trained models in Scene class not implemented yet!")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, dynamic_pcd=scene_info.dynamic_point_cloud,
                                           road_pcd=scene_info.road_pcd, sky_pcd=scene_info.sky_pcd, 
                                           static_vehicle_pcd=scene_info.static_vehicle_pcd,
                                           vehicle_pcd_dict=scene_info.vehicle_pcd_dict,
                                           vehicle_init_pose_dict=scene_info.vehicle_init_pose_dict)

        self.gaussians.aabb = scene_info.cam_frustum_aabb
        self.gaussians.aabb_tensor = torch.tensor(scene_info.cam_frustum_aabb, dtype=torch.float32).cuda()
        self.gaussians.nerf_normalization = scene_info.nerf_normalization
        self.gaussians.img_width = scene_info.train_cameras[0].width
        self.gaussians.img_height = scene_info.train_cameras[0].height
        if scene_info.occ_grid is not None:
            self.gaussians.occ_grid = torch.tensor(scene_info.occ_grid, dtype=torch.bool).cuda() 
        else:
            self.gaussians.occ_grid = scene_info.occ_grid
        self.gaussians.occ_voxel_size = args.occ_voxel_size
        # for deformation-field
        if hasattr(self.gaussians, '_deformation'):
            self.gaussians._deformation.deformation_net.set_aabb(scene_info.cam_frustum_aabb[1],
                                                scene_info.cam_frustum_aabb[0])


    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)


    def save_gridgs(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_grid".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getFullCameras(self, scale=1.0):
        return self.full_cameras[scale]