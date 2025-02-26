# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: seamless_merging.py
# Time: 5/15/24 2:31 PM
# Description: Seamless merging of partitioned point clouds.
import os.path
import json
import numpy as np
from glob import glob
import pickle

import torch
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def extend_inf_x_z_bbox(partition_id, m_region, n_region):
    # bbox: [x_min, x_max, z_min, z_max]
    # m_region and n_region must >= 2
    # Perform seamless merging by determining   the x and z-axis ranges of the original bbox based on the partition's position.
    x, z = int(partition_id.split("_")[0]), int(partition_id.split("_")[1])  # Get the partition's coordinates
    if x == 1 and z == 1:  # Top-left corner
        return [True, False, True, False]
    if x == m_region and z == 1:  # Top-right corner
        return [False, True, True, False]
    if x == 1 and z == n_region:  # Bottom-left corner
        return [True, False, False, True]
    if x == m_region and z == n_region:  # Bottom-right corner
        return [False, True, False, True]
    if 2 <= x <= m_region - 1 and z == 1:  # Top-middle
        return [False, False, True, False]
    if 2 <= z <= n_region - 1 and x == 1:  # Left-middle
        return [True, False, False, False]
    if 2 <= x <= m_region - 1 and z == n_region:  # Bottom-middle
        return [False, False, False, True]
    if 2 <= z <= n_region - 1 and x == m_region:  # Right-middle
        return [False, True, False, False]
    if 2 <= x <= m_region - 1 and 2 <= z <= n_region - 1:  # Middle
        return [False, False, False, False]


def load_ply(path):
    plydata = PlyData.read(path)
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def extract_point_cloud(points, bbox):
    """Filter the point cloud corresponding to a partition based on the camera's boundary."""
    mask = (points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[1]) & (
            points[:, 1] >= bbox[2]) & (points[:, 1] <= bbox[3]) & (
                   points[:, 2] >= bbox[4]) & (points[:, 2] <= bbox[5])  # 筛选在范围内的点云，得到对应的mask
    return mask


def seamless_merge(model_path, partition_point_cloud_dir):
    save_merge_dir = os.path.join(partition_point_cloud_dir, "point_cloud.ply")

    # Load partition data
    with open(os.path.join(model_path, "partition_data.pkl"), "rb") as f:
        partition_scene = pickle.load(f)

    m_region, n_region = 0, 0
    # Determine the number of partitions
    for partition in partition_scene:
        m, n = int(partition.partition_id.split("_")[0]), int(partition.partition_id.split("_")[1])
        if m > m_region: m_region = m
        if n > n_region: n_region = n

    # Iterate through all partition point clouds
    xyz_list = []
    features_dc_list = []
    features_extra_list = []
    opacities_list = []
    scales_list = []
    rots_list = []

    for partition in partition_scene:
        point_cloud_path = os.path.join(partition_point_cloud_dir, f"{partition.partition_id}_point_cloud.ply")
        xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)
        extend_camera_bbox = partition.extend_camera_bbox  # Original camera bounding box
        x_max = extend_camera_bbox[1]
        x_min = extend_camera_bbox[0]
        z_max = extend_camera_bbox[3]
        z_min = extend_camera_bbox[2]

        flag = extend_inf_x_z_bbox(partition.partition_id, m_region, n_region)
        if partition.partition_id == "1_1":
            flag = [True, False, True, True]
        if partition.partition_id == "2_1":
            flag = [False, True, True, True]

        x_max = np.inf if flag[1] else x_max
        x_min = -np.inf if flag[0] else x_min
        z_max = np.inf if flag[3] else z_max
        z_min = -np.inf if flag[2] else z_min
        # x_max = np.inf
        # x_min = -np.inf
        # z_max = np.inf
        # z_min = -np.inf
        print('region:', point_cloud_path)
        print('x_min:{}, x_max:{}, z_min:{}, z_max:{}'.format(x_min, x_max, z_min, z_max))

        point_select_bbox = [x_min, x_max,  # [x_min, x_max, y_min, y_max, z_min, z_max]
                             -np.inf, np.inf,
                             # Use the y-axis range of the original point cloud as the restoration range, as y-axis was not considered during partitioning
                             z_min, z_max]

        mask = extract_point_cloud(xyz, point_select_bbox)
        xyz_list.append(xyz[mask])
        features_dc_list.append(features_dc[mask])
        features_extra_list.append(features_extra[mask])
        opacities_list.append(opacities[mask])
        scales_list.append(scales[mask])
        rots_list.append(rots[mask])

        fig, ax = plt.subplots()
        x_pos = xyz[mask][:, 0]
        z_pos = xyz[mask][:, 2]
        ax.scatter(x_pos, z_pos, c='k', s=1)

        rect = patches.Rectangle((x_min, z_min), x_max-x_min, z_max-z_min, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.title.set_text('Plot of 2D Points')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        fig.tight_layout()
        fig.savefig(os.path.join(partition_point_cloud_dir, f'{partition.partition_id}_pcd.png'), dpi=200)
        plt.close(fig)
        print('point_cloud_path:', point_cloud_path)

    points = np.concatenate(xyz_list, axis=0)
    features_dc_list = np.concatenate(features_dc_list, axis=0)
    features_extra_list = np.concatenate(features_extra_list, axis=0)
    opacities_list = np.concatenate(opacities_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    rots_list = np.concatenate(rots_list, axis=0)

    # Remove duplicate points due to overlapping boundaries
    points, mask = np.unique(points, axis=0, return_index=True)
    features_dc_list = features_dc_list[mask]
    features_extra_list = features_extra_list[mask]
    opacities_list = opacities_list[mask]
    scales_list = scales_list[mask]
    rots_list = rots_list[mask]

    global_model = GaussianModel(3)
    global_params = {'xyz': torch.from_numpy(points).float().cuda(),
                     'rotation': torch.from_numpy(rots_list).float().cuda(),
                     'scaling': torch.from_numpy(scales_list).float().cuda(),
                     'opacity': torch.from_numpy(opacities_list).float().cuda(),
                     'features_dc': torch.from_numpy(features_dc_list).float().cuda().permute(0, 2, 1),
                     'features_rest': torch.from_numpy(features_extra_list).float().cuda().permute(0, 2, 1)}

    global_model.set_params(global_params)
    global_model.save_ply(save_merge_dir)


if __name__ == '__main__':
    seamless_merge("output/train",
                   "output/train/point_cloud/iteration_60000")