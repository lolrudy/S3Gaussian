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

import torch
import gc
import copy 
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
from utils.refs import THING
from pyrotation.conversion import euler_angle_from_quaternion

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self.prune_3d = args.prune_3d
        self.reset_visible = args.reset_visible
        self._xyz = torch.empty(0)
        # self._deformation =  torch.empty(0)
        self._deformation = deform_network(args)
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self._thing_map = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._thing_map,
            self._vehicle_idx_table,
            self.vehicle_gid2idx,
            self.vehicle_idx2gid
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._thing_map,
        self._vehicle_idx_table,
        self.vehicle_gid2idx,
        self.vehicle_idx2gid) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, 
                        dynamic_pcd: BasicPointCloud=None, 
                        road_pcd: BasicPointCloud=None, 
                        sky_pcd: BasicPointCloud=None,
                        static_vehicle_pcd: BasicPointCloud=None,
                        vehicle_pcd_dict: dict=None,
                        vehicle_init_pose_dict: dict=None,
                        vehicle_pcd:BasicPointCloud=None):
        self.spatial_lr_scale = spatial_lr_scale
        self.vehicle_idx2gid = {}
        self.vehicle_gid2idx = {}
        self.vehicle_init_pose_dict = vehicle_init_pose_dict
        if dynamic_pcd is not None:
            if vehicle_pcd_dict is not None:
                fused_vehicle_pts = []
                fused_vehicle_colors = []
                fused_vehicle_idxes = []
                vehicle_idx = 1
                for vehicle_gid in vehicle_pcd_dict.keys():
                    vehicle_pcd = vehicle_pcd_dict[vehicle_gid]
                    vehicle_pts = torch.tensor(np.asarray(vehicle_pcd.points))
                    vehicle_colors = torch.tensor(np.asarray(vehicle_pcd.colors))
                    fused_vehicle_pts.append(vehicle_pts)
                    fused_vehicle_colors.append(vehicle_colors)
                    repeat_vehicle_idx = torch.ones(len(vehicle_pcd.points)) * vehicle_idx
                    fused_vehicle_idxes.append(repeat_vehicle_idx)
                    self.vehicle_idx2gid[vehicle_idx] = vehicle_gid
                    self.vehicle_gid2idx[vehicle_gid] = vehicle_idx
                    vehicle_idx += 1
                    
                fused_vehicle_pts = torch.cat(fused_vehicle_pts).float()
                fused_vehicle_colors = torch.cat(fused_vehicle_colors).float()
                fused_vehicle_idxes = torch.cat(fused_vehicle_idxes).float()
                fused_point_cloud = torch.cat([torch.tensor(np.asarray(pcd.points)), 
                                            torch.tensor(np.asarray(static_vehicle_pcd.points)), 
                                            torch.tensor(np.asarray(road_pcd.points)), 
                                            torch.tensor(np.asarray(sky_pcd.points)), 
                                            torch.tensor(np.asarray(dynamic_pcd.points)),
                                            fused_vehicle_pts]).float().cuda()
                fused_color = RGB2SH(torch.cat([torch.tensor(np.asarray(pcd.colors)), 
                                                torch.tensor(np.asarray(static_vehicle_pcd.colors)), 
                                                torch.tensor(np.asarray(road_pcd.colors)),
                                                torch.tensor(np.asarray(sky_pcd.colors)),
                                                torch.tensor(np.asarray(dynamic_pcd.colors)),
                                                fused_vehicle_colors]).float().cuda())
                static_vehicle_pt_num = static_vehicle_pcd.points.shape[0]
                dynamic_vehicle_pt_num = fused_vehicle_pts.shape[0]
            else:
                if vehicle_pcd is not None:
                    fused_vehicle_pts = torch.tensor(np.asarray(vehicle_pcd.points))
                    fused_vehicle_idxes = torch.zeros(len(fused_vehicle_pts))
                    static_vehicle_pt_num = 0
                    
                    fused_point_cloud = torch.cat([torch.tensor(np.asarray(pcd.points)), 
                                                torch.tensor(np.asarray(road_pcd.points)), 
                                                torch.tensor(np.asarray(sky_pcd.points)), 
                                                torch.tensor(np.asarray(dynamic_pcd.points)),
                                                fused_vehicle_pts]).float().cuda()
                    fused_color = RGB2SH(torch.cat([torch.tensor(np.asarray(pcd.colors)), 
                                                    torch.tensor(np.asarray(road_pcd.colors)),
                                                    torch.tensor(np.asarray(sky_pcd.colors)),
                                                    torch.tensor(np.asarray(dynamic_pcd.colors)),
                                                    torch.tensor(np.asarray(vehicle_pcd.colors))]).float().cuda())
                    dynamic_vehicle_pt_num = fused_vehicle_pts.shape[0]
                else:
                    dynamic_vehicle_pt_num = 0
                    static_vehicle_pt_num = 0
                    fused_point_cloud = torch.cat([torch.tensor(np.asarray(pcd.points)), 
                                                torch.tensor(np.asarray(road_pcd.points)), 
                                                torch.tensor(np.asarray(sky_pcd.points)), 
                                                torch.tensor(np.asarray(dynamic_pcd.points))]).float().cuda()
                    fused_color = RGB2SH(torch.cat([torch.tensor(np.asarray(pcd.colors)), 
                                                    torch.tensor(np.asarray(road_pcd.colors)),
                                                    torch.tensor(np.asarray(sky_pcd.colors)),
                                                    torch.tensor(np.asarray(dynamic_pcd.colors))]).float().cuda())
                    fused_vehicle_idxes = torch.zeros([0])
            static_point_num = pcd.points.shape[0]
            road_point_num = road_pcd.points.shape[0]
            sky_point_num = sky_pcd.points.shape[0]
            dynamic_point_num = dynamic_pcd.points.shape[0]
            print(f'original static point number: {static_point_num+static_vehicle_pt_num}')
            print(f'original dynamic point number: {dynamic_point_num + dynamic_vehicle_pt_num}')
            self._deformation_table = torch.zeros((fused_point_cloud.shape[0]),device="cuda")
            self._deformation_table[-dynamic_point_num-dynamic_vehicle_pt_num:] = 1
            self._deformation_table = self._deformation_table.bool()
            
            self._thing_map =  torch.cat([torch.ones([static_point_num]) * THING.STATIC_OBJECT,
                                          torch.ones([static_vehicle_pt_num]) * THING.VEHICLE,
                                          torch.ones([road_point_num]) * THING.ROAD,
                                          torch.ones([sky_point_num]) * THING.SKY,
                                          torch.ones([dynamic_point_num]) * THING.DYNAMIC_OBJECT,
                                          torch.ones([dynamic_vehicle_pt_num]) * THING.VEHICLE]).int().cuda()
            self._vehicle_idx_table = torch.cat([torch.ones([static_point_num]) * -1,
                                          torch.ones([static_vehicle_pt_num]) * -1,
                                          torch.ones([road_point_num]) * -1,
                                          torch.ones([sky_point_num]) * -1,
                                          torch.ones([dynamic_point_num]) * -1,
                                          fused_vehicle_idxes]).int().cuda()

        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            self._deformation_table = torch.gt(torch.ones((fused_point_cloud.shape[0]),device="cuda"),0)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        if dynamic_pcd is not None:
            flat_mask = (self._thing_map == THING.SKY) | (self._thing_map == THING.ROAD)
            scales[flat_mask, 2] = -100
 
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
            
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.grid_optimizer = torch.optim.Adam([ {'params': list(self._deformation.get_grid_parameters()), 'lr': 0.1, "name": "grid"}],
                                               lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                lr_pos = lr
            if  "grid" in param_group["name"]: # 这里也有
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation": # 这里一开始就会进
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
        return lr_pos

    def construct_list_of_attributes(self, save_color=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if save_color:
            l.extend(['red', 'green', 'blue'])
        else:
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        return l
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz
    # def save_ply_dynamic(path):
    #     for time in range(self._deformation.shape(-1)):
    #         xyz = self.compute_deformation(time)
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._thing_map = torch.zeros((self.get_xyz.shape[0]),device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "thing_map.pth")):
            self._thing_map = torch.load(os.path.join(path, "thing_map.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "vehicle_idx_table.pth")):
            self._vehicle_idx_table = torch.load(os.path.join(path, "vehicle_idx_table.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)
        
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._thing_map,os.path.join(path, "thing_map.pth"))
        torch.save(self._vehicle_idx_table,os.path.join(path, "vehicle_idx_table.pth"))
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def save_ply_split(self, save_dir, dx_list, visibility_filter):
        mkdir_p(save_dir)
        dynamic_pcd_path = os.path.join(save_dir, 'dynamic.ply')
        static_pcd_path = os.path.join(save_dir, 'static.ply')
        dynamic_attributes_list = []  # List to store dynamic attributes
        static_attributes_list = []   # List to store static attributes

        # dynamic and static
        # 如果提供了掩码，仅选择掩码为True的点
        # for dx in dx_list:
        # dx = dx_list [24]
        if True:
            # DO NOT UPDATE XYZ
            # update xyz
            # self._xyz = self._xyz + dx
            mask = self._deformation_table
            dynamic_mask = mask 
            dynamic_mask = dynamic_mask.cpu().numpy() if isinstance(dynamic_mask, torch.Tensor) else dynamic_mask
            dynamic_indices = np.where(dynamic_mask)[0]
            static_mask = ~mask
            static_mask = static_mask.cpu().numpy() if isinstance(static_mask, torch.Tensor) else static_mask
            static_indices = np.where(static_mask)[0]

            # Extract dynamic attributes based on the mask
            dynamic_xyz = self._xyz[dynamic_indices].detach().cpu().numpy()
            dynamic_normals = np.zeros_like(dynamic_xyz)  # Assuming normals are zeros for simplicity
            dynamic_f_dc = self._features_dc[dynamic_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            dynamic_f_rest = self._features_rest[dynamic_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            dynamic_opacities = self._opacity[dynamic_indices].detach().cpu().numpy()
            dynamic_scale = self._scaling[dynamic_indices].detach().cpu().numpy()
            dynamic_rotation = self._rotation[dynamic_indices].detach().cpu().numpy()
            dynamic_rgb = SH2RGB(dynamic_f_dc) * 255.
            dynamic_rgb = dynamic_rgb.clip(0, 255)
            dynamic_rgb = dynamic_rgb.astype(np.int)

            # Extract static attributes based on the mask
            static_xyz = self._xyz[static_indices].detach().cpu().numpy()
            static_normals = np.zeros_like(static_xyz)  # Assuming normals are zeros for simplicity
            static_f_dc = self._features_dc[static_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            static_f_rest = self._features_rest[static_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            static_opacities = self._opacity[static_indices].detach().cpu().numpy()
            static_scale = self._scaling[static_indices].detach().cpu().numpy()
            static_rotation = self._rotation[static_indices].detach().cpu().numpy()
            static_rgb = SH2RGB(static_f_dc) * 255.
            static_rgb = static_rgb.clip(0, 255)
            static_rgb = static_rgb.astype(np.int)
            
            # Append dynamic and static attributes to their respective lists
            # dynamic_attributes_list.append((dynamic_xyz, dynamic_normals, dynamic_f_dc, dynamic_f_rest, dynamic_opacities, dynamic_scale, dynamic_rotation))
            # static_attributes_list.append((static_xyz, static_normals, static_f_dc, static_f_rest, static_opacities, static_scale, static_rotation))
            
            dynamic_attributes_list.append((dynamic_xyz, dynamic_normals, dynamic_rgb))
            static_attributes_list.append((static_xyz, static_normals, static_rgb))

        # Concatenate dynamic attributes after the loop
        concatenated_dynamic_attributes = [np.concatenate(attr, axis=0) for attr in zip(*dynamic_attributes_list)]

        # Concatenate static attributes after the loop
        concatenated_static_attributes = [np.concatenate(attr, axis=0) for attr in zip(*static_attributes_list)]
  
        # Prepare PlyData for dynamic point cloud
        dtype_full = []
        for attribute in self.construct_list_of_attributes(save_color=True):
            new_item = (attribute, 'f4') if attribute not in ('red', 'green', 'blue') else (attribute, 'u1')
            dtype_full.append(new_item)
        dynamic_elements = np.empty(concatenated_dynamic_attributes[0].shape[0], dtype=dtype_full)
        dynamic_attributes = np.concatenate(concatenated_dynamic_attributes, axis=1)
        dynamic_elements[:] = list(map(tuple, dynamic_attributes))
        dynamic_el = PlyElement.describe(dynamic_elements, 'vertex')

        # Write dynamic PlyData to file
        PlyData([dynamic_el]).write(dynamic_pcd_path)

        # Prepare PlyData for static point cloud
        static_elements = np.empty(concatenated_static_attributes[0].shape[0], dtype=dtype_full)
        static_attributes = np.concatenate(concatenated_static_attributes, axis=1)
        static_elements[:] = list(map(tuple, static_attributes))
        static_el = PlyElement.describe(static_elements, 'vertex')

        # Write static PlyData to file
        PlyData([static_el]).write(static_pcd_path)
                    
    def reset_opacity(self):
        # ONLY RESET OPACITY FOR VISIBLE POINTS, NOT WORKING SINCE MAXRADII2D IS SET TO 0 IN DENSIFICATION
        if self.reset_visible:
            visible_point_mask = self.max_radii2D > 1e-3
        else:
            visible_point_mask = torch.ones_like(self.max_radii2D).bool()
        opacities_new = self._opacity.clone()
        opacities_new[visible_point_mask] = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))[visible_point_mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self._thing_map = self._thing_map[valid_points_mask]
        self._vehicle_idx_table = self._vehicle_idx_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
                              new_deformation_table, new_thing_map, new_vehicle_idx_table):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self._thing_map = torch.cat([self._thing_map,new_thing_map],-1)
        self._vehicle_idx_table = torch.cat([self._vehicle_idx_table, new_vehicle_idx_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.max_radii2D.shape[0] < self.get_xyz.shape[0]:
            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((self.get_xyz.shape[0]-self.max_radii2D.shape[0]), device="cuda")])

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        new_thing_map = self._thing_map[selected_pts_mask].repeat(N)
        new_vehicle_idx_table = self._vehicle_idx_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
                                   new_deformation_table, new_thing_map, new_vehicle_idx_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 主动增加稀疏点云
        # if not hasattr(self,"voxel_size"):
        #     self.voxel_size = 8  
        # if not hasattr(self,"density_threshold"):
        #     self.density_threshold = density_threshold
        # if not hasattr(self,"displacement_scale"):
        #     self.displacement_scale = displacement_scale
        # point_cloud = self.get_xyz.detach().cpu()
        # sparse_point_mask = self.downsample_point(point_cloud)
        # _, low_density_points, new_points, low_density_index = addpoint(point_cloud[sparse_point_mask],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        # sparse_point_mask = sparse_point_mask.to(grads_accum_mask)
        # low_density_index = low_density_index.to(grads_accum_mask)
        # if new_points.shape[0] < 100 :
        #     self.density_threshold /= 2
        #     self.displacement_scale /= 2
        #     print("reduce diplacement_scale to: ",self.displacement_scale)
        # global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(grads_accum_mask)
        # global_mask[sparse_point_mask] = low_density_index
        # selected_pts_mask_grow = torch.logical_and(global_mask, grads_accum_mask)
        # print("降采样点云:",sparse_point_mask.sum(),"选中的稀疏点云：",global_mask.sum(),"梯度累计点云：",grads_accum_mask.sum(),"选中增长点云：",selected_pts_mask_grow.sum())
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # breakpoint()        
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        new_thing_map = self._thing_map[selected_pts_mask]
        new_vehicle_idx_table = self._vehicle_idx_table[selected_pts_mask]
        # if opt.add_point:
        # selected_xyz, grow_xyz = self.add_point_by_mask(selected_pts_mask_grow.to(self.get_xyz.device), self.displacement_scale)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
                                   new_deformation_table, new_thing_map, new_vehicle_idx_table)
        # print("被动增加点云：",selected_xyz.shape[0])
        # print("主动增加点云：",selected_pts_mask.sum())
        # if model_path is not None and iteration is not None:
        #     point = combine_pointcloud(self.get_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy(), selected_xyz.detach().cpu().numpy())
        #     write_path = os.path.join(model_path,"add_point_cloud")
        #     os.makedirs(write_path,exist_ok=True)
        #     o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        #     print("write output.")
    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    
        # while (mask_d.sum()/final_point.shape[0])<0.5:
        #     perturb/=2
        #     displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        #     final_point = selected_point + displacements
        #     mask_a = final_point<xyz_max 
        #     mask_b = final_point>xyz_min
        #     mask_c = mask_a & mask_b
        #     mask_d = mask_c.all(dim=1)
        #     final_point = final_point[mask_d]
        return final_point, mask_d    
    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)
        # displacements = torch.randn(selected_xyz.shape[0], 3).to(self._xyz) * perturb

        # new_xyz = selected_xyz + displacements
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]
        new_thing_map = self._thing_map[selected_pts_mask][mask]
        new_vehicle_idx_table = self._vehicle_idx_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, 
                                   new_rotation, new_deformation_table, new_thing_map, new_vehicle_idx_table)
        return selected_xyz, new_xyz
    
    # def downsample_point(self, point_cloud):
    #     if not hasattr(self,"voxel_size"):
    #         self.voxel_size = 8  
    #     point_downsample = point_cloud
    #     flag = False 
    #     while point_downsample.shape[0]>1000:
    #         if flag:
    #             self.voxel_size+=8
    #         point_downsample = downsample_point_cloud_open3d(point_cloud,voxel_size=self.voxel_size)
    #         flag = True
    #     print("point size:",point_downsample.shape[0])
    #     # downsampled_point_mask = torch.eq(point_downsample.view(1,-1,3), point_cloud.view(-1,1,3)).all(dim=1)
    #     downsampled_point_index = find_indices_in_A(point_cloud, point_downsample)
    #     downsampled_point_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(point_downsample.device)
    #     downsampled_point_mask[downsampled_point_index]=True
    #     return downsampled_point_mask
    
    # def grow(self, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
    #     if not hasattr(self,"voxel_size"):
    #         self.voxel_size = 8  
    #     if not hasattr(self,"density_threshold"):
    #         self.density_threshold = density_threshold
    #     if not hasattr(self,"displacement_scale"):
    #         self.displacement_scale = displacement_scale
    #     flag = False
    #     point_cloud = self.get_xyz.detach().cpu()
    #     point_downsample = point_cloud.detach()
    #     downsampled_point_index = self.downsample_point(point_downsample)


    #     _, low_density_points, new_points, low_density_index = addpoint(point_cloud[downsampled_point_index],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
    #     if new_points.shape[0] < 100 :
    #         self.density_threshold /= 2
    #         self.displacement_scale /= 2
    #         print("reduce diplacement_scale to: ",self.displacement_scale)

    #     elif new_points.shape[0] == 0:
    #         print("no point added")
    #         return
    #     global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool)

    #     global_mask[downsampled_point_index] = low_density_index
    #     global_mask
    #     selected_xyz, new_xyz = self.add_point_by_mask(global_mask.to(self.get_xyz.device), self.displacement_scale)
    #     print("point growing,add point num:",global_mask.sum())
    #     if model_path is not None and iteration is not None:
    #         point = combine_pointcloud(point_cloud, selected_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy())
    #         write_path = os.path.join(model_path,"add_point_cloud")
    #         os.makedirs(write_path,exist_ok=True)
    #         o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
    #     return
    
    def prune(self, max_grad, min_opacity, extent, max_screen_size, prune_dynamic=False):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            if self.prune_3d:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(prune_mask, big_points_ws)

        if not prune_dynamic:
            prune_mask = prune_mask & (~self._deformation_table)
            
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)

    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def compute_flat_regulation(self, ):
        flat_mask = (self._thing_map == THING.ROAD) | (self._thing_map == THING.SKY)
        rot_flat = self.get_rotation[flat_mask]
        euler_flat = euler_angle_from_quaternion(rot_flat)
        scale_flat = self.get_scaling[flat_mask]
        rot_reg_loss = torch.abs(euler_flat[:, 0]).mean() + torch.abs(euler_flat[:, 1]).mean()
        scale_reg_loss = scale_flat[:, 2].mean()
        return rot_reg_loss + scale_reg_loss


def merge_models(old_model:GaussianModel, new_model:GaussianModel,hyper,merged_model=None):
    # 创建一个新的模型实例，这里我们假设max_sh_degree和hyper可以从任一模型获取

    if merged_model is None:
        merged_model = GaussianModel(old_model.max_sh_degree, hyper)

    # 合并时需要的参数化属性列表
    parameterized_attributes = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']

    # 遍历模型的属性
    for attr in new_model.__dict__.keys():
        attr_value_static = getattr(old_model, attr)
        attr_value_dynamic = getattr(new_model, attr)
        if isinstance(attr_value_dynamic, torch.Tensor):
            # 检查是否是参数化属性
            if attr in parameterized_attributes:
                # 创建一个合并后的tensor，其中包含动态和静态部分
                if isinstance(attr_value_static.data, torch.Tensor) and isinstance(attr_value_dynamic.data, torch.Tensor):
                    combined_data = torch.cat([attr_value_dynamic.data, attr_value_static.data], dim=0)
                    setattr(merged_model, attr, torch.nn.Parameter(combined_data))
            else:
                # 非参数化属性但是是tensor，同样合并
                combined_data = torch.cat([attr_value_dynamic, attr_value_static], dim=0)
                setattr(merged_model, attr, combined_data)
            del attr_value_dynamic
            del attr_value_static
            gc.collect()
            torch.cuda.empty_cache() 
        elif attr == '_deformation':
            
            setattr(merged_model, attr, copy.deepcopy(attr_value_dynamic))
            del attr_value_dynamic
            del attr_value_static
            gc.collect()
            torch.cuda.empty_cache()     

        elif attr == 'optimizer':
            continue

        else:
            # 非tensor属性，假设静态和动态模型中是相同的
            setattr(merged_model, attr, attr_value_dynamic)
    return merged_model

