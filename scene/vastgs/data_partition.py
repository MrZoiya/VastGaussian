# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: data_partition.py
# Time: 5/15/24 2:28 PM
# Des: 数据划分策略
"""
Because it is not understood how to perform a Manhattan world alignment so that the y-axis of the world coordinates is
perpendicular to the ground plane, this implementation assumes that the coordinates are already for its"""
import copy
import os
import numpy as np
from typing import NamedTuple
import pickle
import math

from scene.dataset_readers import CameraInfo, storePly
from utils.graphics_utils import BasicPointCloud
from scene.vastgs.graham_scan import run_graham_scan
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z] 坐标


class CameraPartition(NamedTuple):
    partition_id: str  # Partial character numbering
    cameras: list  # All cameras corresponding to this section CameraPose
    point_cloud: BasicPointCloud  # The point cloud corresponding to this part
    ori_camera_bbox: list  # 边界拓展前相机围成的相机 边界坐标 [x_min, x_max, z_min, z_max]，同时方便根据原始的边框对最后训练出来的点云进行裁减，获取原始的点云范围
    extend_camera_bbox: list  # 按照extend_rate拓展后 相机围成的边界坐标
    extend_rate: float  # 边界拓展的比例 默认0.2

    ori_camera_bbox: list  # Box enclosed by cameras before boundary expansion Boundary coordinates [x_min, x_max, z_min, z_max], while facilitating the cropping of the final trained point cloud based on the original borders to obtain the original point cloud extent
    extend_point_bbox: list  # Points filtered by the extended camera boundaries of these points


class ProgressiveDataPartitioning:
    def __init__(self, scene_info, train_cameras, model_path, m_region=2, n_region=4, extend_rate=0.2,
                 visible_rate=0.25):
        self.partition_scene = None
        self.pcd = scene_info.point_cloud
        # print(f"self.pcd={self.pcd}")
        self.model_path = model_path  # Store model location
        self.partition_dir = os.path.join(model_path, "partition_point_cloud")
        self.partition_ori_dir = os.path.join(self.partition_dir, "ori")
        self.partition_extend_dir = os.path.join(self.partition_dir, "extend")
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        self.m_region = m_region
        self.n_region = n_region
        self.extend_rate = extend_rate
        self.visible_rate = visible_rate

        if not os.path.exists(self.partition_ori_dir): os.makedirs(self.partition_ori_dir)  # Create a folder for post-chunking, pre-expansion point clouds.
        if not os.path.exists(self.partition_extend_dir): os.makedirs(self.partition_extend_dir)  # Create a folder for chunked, expanded, and point clouds.
        if not os.path.exists(self.partition_visible_dir): os.makedirs(
            self.partition_visible_dir)  # Creating a folder for storing chunks after the visibility camera has been selected and the point cloud has been selected.
        self.fig, self.ax = self.draw_pcd(self.pcd, train_cameras)
        self.run_DataPartition(train_cameras)

    def draw_pcd(self, pcd, train_cameras):        
        x_coords = pcd.points[:, 0]
        z_coords = pcd.points[:, 2]
        fig, ax = plt.subplots()
        ax.scatter(x_coords, z_coords, c=(pcd.colors), s=1)
        ax.title.set_text('Plot of 2D Points')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, 'pcd.png'),dpi=200)
        x_coords = np.array([cam.camera_center[0].item() for cam in train_cameras])
        z_coords = np.array([cam.camera_center[2].item() for cam in train_cameras])
        ax.scatter(x_coords, z_coords, color='red', s=1)
        fig.savefig(os.path.join(self.model_path, 'camera_on_pcd.png'),dpi=200)
        return fig, ax
        
    def draw_partition(self, partition_list):
        for partition in partition_list:
            ori_bbox = partition.ori_camera_bbox
            extend_bbox = partition.extend_camera_bbox
            x_min, x_max, z_min, z_max = ori_bbox
            ex_x_min, ex_x_max, ex_z_min, ex_z_max = extend_bbox
            rect_ori = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, linewidth=1, edgecolor='blue',
                                         facecolor='none')
            rect_ext = patches.Rectangle((ex_x_min, ex_z_min), ex_x_max-ex_x_min, ex_z_max-ex_z_min, linewidth=1, edgecolor='y', facecolor='none')
            self.ax.add_patch(rect_ori)
            self.ax.text(x=rect_ori.get_x(), y=rect_ori.get_y(), s=f"{partition.partition_id}", color='black', fontsize=12)
            self.ax.add_patch(rect_ext)
        self.fig.savefig(os.path.join(self.model_path, f'regions.png'),dpi=200)
        return
        
    def run_DataPartition(self, train_cameras):
        if not os.path.exists(self.save_partition_data_dir):
            partition_dict = self.Camera_position_based_region_division(train_cameras)
            partition_dict, refined_ori_bbox = self.refine_ori_bbox(partition_dict)
            # partition_dict, refined_ori_bbox = self.refine_ori_bbox_average(partition_dict)
            partition_list = self.Position_based_data_selection(partition_dict, refined_ori_bbox)
            self.draw_partition(partition_list)
            self.partition_scene = self.Visibility_based_camera_selection(
                partition_list)  # Outputs the scene after visibility filtering, including cameras and point clouds
            self.save_partition_data()
        else:
            self.partition_scene = self.load_partition_data()


    def save_partition_data(self):
        """Serialize and save the partitioned data for easy loading next time."""
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partition_scene, f)

    def load_partition_data(self):
        """Load the partitioned data."""
        with open(self.save_partition_data_dir, 'rb') as f:
            partition_scene = pickle.load(f)
        return partition_scene

    def refine_ori_bbox_average(self, partition_dict):
        """Refine the original bounding boxes to make the boundaries seamless, facilitating seamless merging later.
        The average of the bounding boxes of two adjacent cameras is used as the boundary for seamless merging.
        """
        bbox_with_id = {}
        # 1. Get the camera boundaries for each partition
        for partition_idx, cameras in partition_dict.items():
            # TODO: Needs modification. The original boundary should use the boundary from partitioning,
            #  not the camera positions, to achieve seamless merging.
            camera_list = cameras["camera_list"]
            min_x, max_x = min(camera.pose[0] for camera in camera_list), max(
                camera.pose[0] for camera in camera_list) # min_x, max_x represent the length of the area enclosed by cameras along the x-axis
            min_z, max_z = min(camera.pose[2] for camera in camera_list), max(camera.pose[2] for camera in camera_list)
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox

        # 2. Adjust the camera boundaries along the x-axis first, using the average of the boundary coordinates of two adjacent partitions as the shared boundary
        for m in range(1, self.m_region+1):
            for n in range(1, self.n_region+1):
                if n+1 == self.n_region+1:
                    # Exit if it's the last block
                    break
                partition_idx_1 = str(m) + '_' + str(n)    # Left block
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n+1)  # Right block
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_z = (max_z_1 + min_z_2) / 2
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, min_z_1, mid_z]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, mid_z, max_z_2]

        # 3. Adjust the camera boundaries along the z-axis. Since the adjustments were made along the x-axis,
        # the left and right blocks need to be evenly divided along the x-axis.
        # First, find the maximum x_max of the left partition and the minimum x_min of the right partition
        for m in range(1, self.m_region + 1):
            if m + 1 == self.m_region + 1:
                # Exit if it's the last block
                break
            max_x_left = -np.inf
            min_x_right = np.inf
            for n in range(1, self.n_region+1):   # Left partition
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if max_x > max_x_left: max_x_left = max_x

            for n in range(1, self.n_region+1):   # Right partition
                partition_idx = str(m+1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if min_x < min_x_right: min_x_right = min_x

            # Adjust the boundaries of the left and right partitions
            for n in range(1, self.n_region+1):   # Left partition
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [min_x, mid_x, min_z, max_z]

            for n in range(1, self.n_region + 1):  # Right partition
                partition_idx = str(m+1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [mid_x, max_x, min_z, max_z]

        new_partition_dict = {f"{partition_id}": cameras["camera_list"] for partition_id, cameras in
                              partition_dict.items()}
        return new_partition_dict, bbox_with_id

    def refine_ori_bbox(self, partition_dict):
        """Use continuous camera coordinates as boundaries for seamless partitioning."""
        bbox_with_id = {}
        for partition_idx, cameras in partition_dict.items():
            # TODO: Needs modification. The original boundary should use the boundary from partitioning,
            #  not the camera positions, to achieve seamless merging.
            camera_list = cameras["camera_list"]
            min_x, max_x = min(camera.pose[0] for camera in camera_list), max(
                camera.pose[0] for camera in
                camera_list)  # min_x, max_x represent the length of the area enclosed by cameras along the x-axis
            min_z, max_z = min(camera.pose[2] for camera in camera_list), max(camera.pose[2] for camera in camera_list)
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox

        # 2. Adjust the camera boundaries along the z-axis
        for m in range(1, self.m_region + 1):
            for n in range(1, self.n_region + 1):
                if n + 1 == self.n_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n + 1)  # Upper block
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n)  # Lower block
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_2]["z_mid_camera"].pose
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, mid_z, max_z_1]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, min_z_2, mid_z]

        # 3. Adjust the camera boundaries along the x-axis
        for n in range(1, self.n_region + 1):
            for m in range(1, self.m_region + 1):
                if m + 1 == self.m_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n)  # Left block
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m + 1) + '_' + str(n)  # Right block
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_1]["x_mid_camera"].pose
                bbox_with_id[partition_idx_1] = [min_x_1, mid_x, min_z_1, max_z_1]
                bbox_with_id[partition_idx_2] = [mid_x, max_x_2, min_z_2, max_z_2]

        new_partition_dict = {f"{partition_id}": cameras["camera_list"] for partition_id, cameras in
                              partition_dict.items()}
        return new_partition_dict, bbox_with_id

    def Camera_position_based_region_division(self, train_cameras):
        """1. Region division based on camera positions.
        Approach:
            1. First, project the camera coordinates of the entire scene onto the xz-plane.
            2. Divide all cameras into `m` parts along the x-axis.
            3. Divide each part into `n` parts along the z-axis (by default, the region is divided into 2x4=8 parts),
               while ensuring a balanced number of cameras in each of the `m x n` parts.
            4. Return the boundary coordinates of each part and the cameras corresponding to each part.
        """
        m, n = self.m_region, self.n_region  # m=2, n=4
        CameraPose_list = []
        camera_centers = []
        for idx, camera in enumerate(train_cameras):
            pose = np.array(camera.camera_center.cpu())
            camera_centers.append(pose)
            CameraPose_list.append(
                CameraPose(camera=camera, pose=pose))  # Camera center coordinates in the world coordinate system

        # Save camera coordinates for visualizing camera positions
        storePly(os.path.join(self.partition_dir, 'camera_centers.ply'), np.array(camera_centers),
                 np.zeros_like(np.array(camera_centers)))

        # 2. Divide cameras into `m` parts along the x-axis
        m_partition_dict = {}
        total_camera = len(CameraPose_list)  # Get the total number of cameras
        num_of_camera_per_m_partition = total_camera // m  # Number of cameras per `m` partition
        sorted_CameraPose_by_x_list = sorted(CameraPose_list, key=lambda x: x.pose[0])  # Sort by x-axis coordinates
        # print(sorted_CameraPose_by_x_list)
        for i in range(m):  # Divide all cameras into `m` parts along the x-axis
            m_partition_dict[str(i + 1)] = {"camera_list": sorted_CameraPose_by_x_list[
                                                           i * num_of_camera_per_m_partition:(
                                                    i + 1) * num_of_camera_per_m_partition]}
            if i != m - 1:
                m_partition_dict[str(i + 1)].update({"x_mid_camera": sorted_CameraPose_by_x_list[(
                                                    i + 1) * num_of_camera_per_m_partition - 1]})  # Use the camera of the left block as the seamless boundary
            else:
                m_partition_dict[str(i + 1)].update({"x_mid_camera": None})  # The last block does not need a mid_camera
        if total_camera % m != 0:  # If the number of cameras is not a multiple of `m`, add the remaining cameras to the last part
            m_partition_dict[str(m)]["camera_list"].extend(
                sorted_CameraPose_by_x_list[m * num_of_camera_per_m_partition:])

        # 3. Divide cameras into `n` parts along the z-axis
        partition_dict = {}  # Store the number of cameras in each `m x n` part
        for partition_idx, cameras in m_partition_dict.items():
            partition_total_camera = len(cameras["camera_list"])  # Number of cameras in each `m` part
            num_of_camera_per_n_partition = partition_total_camera // n  # Number of cameras per `n` partition
            sorted_CameraPose_by_z_list = sorted(cameras["camera_list"],
                                                 key=lambda x: x.pose[2])  # Sort by z-axis coordinates
            for i in range(n):  # Divide all cameras into `n` parts along the z-axis
                partition_dict[f"{partition_idx}_{i + 1}"] = {"camera_list": sorted_CameraPose_by_z_list[
                                                                             i * num_of_camera_per_n_partition:(
                                                                                                                           i + 1) * num_of_camera_per_n_partition]}
                if i != n - 1:
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"x_mid_camera": cameras["x_mid_camera"]})
                    partition_dict[f"{partition_idx}_{i + 1}"].update(
                        {"z_mid_camera": sorted_CameraPose_by_z_list[(i + 1) * num_of_camera_per_n_partition - 1]})
                else:
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"x_mid_camera": cameras["x_mid_camera"]})
                    partition_dict[f"{partition_idx}_{i + 1}"].update(
                        {"z_mid_camera": None})  # The last block does not need a mid_camera
            if partition_total_camera % n != 0:  # If the number of cameras is not a multiple of `n`, add the remaining cameras to the last part
                partition_dict[f"{partition_idx}_{n}"]["camera_list"].extend(
                    sorted_CameraPose_by_z_list[n * num_of_camera_per_n_partition:])

        return partition_dict

    def extract_point_cloud(self, pcd, bbox):
        """Extract the point cloud corresponding to a partition from the initial point cloud based on the camera's boundary."""
        mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
                pcd.points[:, 2] >= bbox[2]) & (pcd.points[:, 2] <= bbox[3])  # Filter points within the range to get the corresponding mask
        points = pcd.points[mask]
        colors = pcd.colors[mask]
        normals = pcd.normals[mask]
        return points, colors, normals

    def get_point_range(self, points):
        """Get the x, y, z boundaries of the current point cloud."""
        x_list = points[:, 0]
        y_list = points[:, 1]
        z_list = points[:, 2]
        # print(points.shape)
        return [min(x_list), max(x_list),
                min(y_list), max(y_list),
                min(z_list), max(z_list)]

    def Position_based_data_selection(self, partition_dict, refined_ori_bbox):
        """
        2. Position-based data selection
        Approach:
            1. Calculate the x and z boundaries for each partition.
            2. Extend the boundary coordinates of each partition using `extend_rate` to get new boundary coordinates [x_min, x_max, z_min, z_max].
            3. Extract the point cloud corresponding to the extended boundary.
        Issue: There might still be some high-quality point clouds not selected after determining the bounding box based on cameras.
               Therefore, `extend_rate` is a hyperparameter that needs to be adjusted based on the actual situation.
        :return partition_list: Point clouds, cameras, and boundaries for each partition.
        """
        # Calculate the extended boundary coordinates for each partition and extract the corresponding point cloud
        pcd = self.pcd
        partition_list = []
        point_num = 0
        point_extend_num = 0
        for partition_idx, camera_list in partition_dict.items():
            min_x, max_x, min_z, max_z = refined_ori_bbox[partition_idx]
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            extend_camera_bbox = [min_x - self.extend_rate * (max_x - min_x),
                                  max_x + self.extend_rate * (max_x - min_x),
                                  min_z - self.extend_rate * (max_z - min_z),
                                  max_z + self.extend_rate * (max_z - min_z)]
            print("Partition", partition_idx, "ori_camera_bbox", ori_camera_bbox, "\textend_camera_bbox",
                  extend_camera_bbox)
            ori_camera_centers = []
            for camera_pose in camera_list:
                ori_camera_centers.append(camera_pose.pose)

            # Save original camera positions
            storePly(os.path.join(self.partition_ori_dir, f'{partition_idx}_camera_centers.ply'),
                     np.array(ori_camera_centers),
                     np.zeros_like(np.array(ori_camera_centers)))

            # TODO: Need to add cameras based on the extended boundary
            new_camera_list = []
            extend_camera_centers = []
            for id, camera_list in partition_dict.items():
                for camera_pose in camera_list:
                    if extend_camera_bbox[0] <= camera_pose.pose[0] <= extend_camera_bbox[1] and extend_camera_bbox[
                        2] <= camera_pose.pose[2] <= extend_camera_bbox[3]:
                        extend_camera_centers.append(camera_pose.pose)
                        new_camera_list.append(camera_pose)

            # Save newly added camera positions after extension
            storePly(os.path.join(self.partition_extend_dir, f'{partition_idx}_camera_centers.ply'),
                     np.array(extend_camera_centers),
                     np.zeros_like(np.array(extend_camera_centers)))

            # Extract the corresponding point cloud for this partition
            points, colors, normals = self.extract_point_cloud(pcd,
                                                               ori_camera_bbox)  # Extract point cloud within the original boundary
            points_extend, colors_extend, normals_extend = self.extract_point_cloud(pcd,
                                                                                    extend_camera_bbox)  # Extract point cloud within the extended boundary
            # The paper mentions that the height of the bounding box for the point cloud is selected as the distance from the highest point to the ground plane.
            # However, in this implementation, since the ground plane position is uncertain (in visualization, the ground plane does not align with the xz-axis),
            # the bounding box of the entire point cloud is used as the spatial-aware boundary box.
            partition_list.append(CameraPartition(partition_id=partition_idx, cameras=new_camera_list,
                                                  point_cloud=BasicPointCloud(points_extend, colors_extend,
                                                                              normals_extend),
                                                  ori_camera_bbox=ori_camera_bbox,
                                                  extend_camera_bbox=extend_camera_bbox,
                                                  extend_rate=self.extend_rate,
                                                  ori_point_bbox=self.get_point_range(points),
                                                  extend_point_bbox=self.get_point_range(points_extend),
                                                  ))

            point_num += points.shape[0]
            point_extend_num += points_extend.shape[0]
            storePly(os.path.join(self.partition_ori_dir, f"{partition_idx}.ply"), points,
                     colors)  # Save point cloud before extension
            storePly(os.path.join(self.partition_extend_dir, f"{partition_idx}_extend.ply"), points_extend,
                     colors_extend)  # Save point cloud after extension

        # Before extending the boundary: The amount of data after position-based selection will be much smaller than the initial point cloud,
        # because the boundary formed by the cameras is smaller than the actual boundary. Therefore, using these boundaries to filter the point cloud reduces the number of points.
        # After extending the boundary: Since there will be many overlapping points, the number of points will increase.
        print(f"Total ori point number: {pcd.points.shape[0]}\n", f"Total before extend point number: {point_num}\n",
              f"Total extend point number: {point_extend_num}\n")

        return partition_list


    def get_8_corner_points(self, bbox):
        """Generate the coordinates of the 8 corner points of a bounding box based on the point cloud's boundaries.
        :param bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        return {
            "minx_miny_minz": [x_min, y_min, z_min],  # 1
            "minx_miny_maxz": [x_min, y_min, z_max],  # 2
            "minx_maxy_minz": [x_min, y_max, z_min],  # 3
            "minx_maxy_maxz": [x_min, y_max, z_max],  # 4
            "maxx_miny_minz": [x_max, y_min, z_min],  # 5
            "maxx_miny_maxz": [x_max, y_min, z_max],  # 6
            "maxx_maxy_minz": [x_max, y_max, z_min],  # 7
            "maxx_maxy_maxz": [x_max, y_max, z_max]   # 8
        }

    def point_in_image(self, camera, points):
        """Project points onto a 2D plane using the projection matrix."""
        # Get the coordinates of points on the image plane
        R = camera.R
        T = camera.T
        w2c = np.eye(4)
        w2c[:3, :3] = np.transpose(R)
        w2c[:3, 3] = T
        fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))

        # This implementation might be incorrect, but no obvious errors were observed during experiments. However, it has been fixed.
        # intrinsic_matrix = np.array([
        #    [fx, 0, camera.image_height // 2],
        #    [0, fy, camera.image_width // 2],
        #    [0, 0, 1]
        # ])

        # Fix bug
        intrinsic_matrix = np.array([
            [fx, 0, camera.image_width // 2],
            [0, fy, camera.image_height // 2],
            [0, 0, 1]
        ])

        # Transform points from world coordinates to camera coordinates
        points_camera = np.dot(w2c[:3, :3], points.T) + w2c[:3, 3:].reshape(3, 1)  # [3, n]
        points_camera = points_camera.T  # [n, 3]
        points_camera = points_camera[
            np.where(points_camera[:, 2] > 0)]  # [n, 3] Filter points based on z-axis (positive depth)

        # Project points onto the image plane
        points_image = np.dot(intrinsic_matrix, points_camera.T)  # [3, n]
        points_image = points_image[:2, :] / points_image[2, :]  # [2, n] Normalize by z (perspective division)
        points_image = points_image.T  # [n, 2]

        # Create a mask for points that lie within the image boundaries
        mask = np.where(np.logical_and.reduce((
            points_image[:, 0] >= 0,
            points_image[:, 0] < camera.image_height,
            points_image[:, 1] >= 0,
            points_image[:, 1] < camera.image_width
        )))[0]

        return points_image, points_image[mask], mask

    def Visibility_based_camera_selection(self, partition_list):
        """3. Visibility-based camera selection and coverage-based point selection.
        Approach: Introduce airspace-aware visibility calculation.
            1. Assume the current partition is i, and select cameras from partition j.
            2. Project the bounding box of partition i onto the cameras in partition j, and calculate the area of the projected region
               (the bounding box only considers the ground part, and both original and extended bounding boxes can be discussed).
            3. Calculate the ratio of the projected area to the image pixel area as the visibility.
            4. Add cameras from partition j with visibility greater than the threshold to partition i.
            5. Add all point clouds from partition j that can be projected onto camera s to partition i.
        :param visible_rate: Visibility threshold, default is 0.25, same as in the paper.
        """
        # Create a deep copy of the partition list to avoid modifying the original data
        add_visible_camera_partition_list = copy.deepcopy(partition_list)
        client = 0
        for idx, partition_i in enumerate(partition_list):  # Process partition i
            new_points = []  # Initialize empty arrays to store new points
            new_colors = []
            new_normals = []

            pcd_i = partition_i.point_cloud
            partition_id_i = partition_i.partition_id  # Get the current partition ID
            # Get the 8 corner points of the bounding box for the current partition's point cloud
            partition_ori_point_bbox = partition_i.ori_point_bbox
            partition_extend_point_bbox = partition_i.extend_point_bbox
            ori_8_corner_points = self.get_8_corner_points(
                partition_ori_point_bbox)  # Get the 8 corner points of the original bounding box
            extent_8_corner_points = self.get_8_corner_points(partition_extend_point_bbox)

            corner_points = []
            for point in extent_8_corner_points.values():
                corner_points.append(point)
            storePly(os.path.join(self.partition_extend_dir, f'{partition_id_i}_corner_points.ply'),
                     np.array(corner_points),
                     np.zeros_like(np.array(corner_points)))

            total_partition_camera_count = 0  # Count of cameras in the current partition
            for partition_j in partition_list:  # Process partition j
                partition_id_j = partition_j.partition_id  # Get the current partition ID
                if partition_id_i == partition_id_j: continue  # Skip if the current partition is the same as partition i
                print(f"Now processing partition i:{partition_id_i} and j:{partition_id_j}")
                # Get the point cloud of the current partition
                pcd_j = partition_j.point_cloud

                append_camera_count = 0  # Count of cameras added from partition j
                # Iterate through each camera in partition j
                for cameras_pose in partition_j.cameras:
                    camera = cameras_pose.camera  # Get the current camera
                    # Project the bounding box of partition i onto the current camera
                    proj_8_corner_points = {}
                    for key, point in extent_8_corner_points.items():
                        points_in_image, _, _ = self.point_in_image(camera, np.array([point]))
                        if len(points_in_image) == 0: continue
                        proj_8_corner_points[key] = points_in_image[0]

                    # Coverage-based point selection
                    # Calculate the ratio of the projected area to the image area
                    if not len(list(proj_8_corner_points.values())) > 3: continue
                    pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)
                    if pkg["intersection_rate"] >= self.visible_rate:
                        collect_names = [camera_pose.camera.image_name for camera_pose in
                                         add_visible_camera_partition_list[idx].cameras]
                        if cameras_pose.camera.image_name in collect_names:
                            continue  # Skip if the camera already exists
                        append_camera_count += 1
                        # Add the current camera from partition j to partition i
                        add_visible_camera_partition_list[idx].cameras.append(cameras_pose)
                        # Select points from partition j that can be projected onto the current camera
                        _, _, mask = self.point_in_image(camera,
                                                         pcd_j.points)  # Points to be added from the original point cloud
                        updated_points, updated_colors, updated_normals = pcd_j.points[mask], pcd_j.colors[mask], \
                        pcd_j.normals[mask]
                        # Update the new points for partition i, ensuring no duplicates
                        new_points.append(updated_points)
                        new_colors.append(updated_colors)
                        new_normals.append(updated_normals)

                        with open(os.path.join(self.model_path, "graham_scan"), 'a') as f:
                            f.write(f"intersection_area:{pkg['intersection_area']}\t"
                                    f"image_area:{pkg['image_area']}\t"
                                    f"intersection_rate:{pkg['intersection_rate']}\t"
                                    f"partition_i:{partition_id_i}\t"
                                    f"partition_j:{partition_id_j}\t"
                                    f"append_camera_id:{camera.image_name}\t"
                                    f"append_camera_count:{append_camera_count}\n")
                total_partition_camera_count += append_camera_count

            with open(os.path.join(self.model_path, "partition_cameras"), 'a') as f:
                f.write(f"partition_id:{partition_id_i}\t"
                        f"total_append_camera_count:{total_partition_camera_count}\t"
                        f"total_camera:{len(add_visible_camera_partition_list[idx].cameras)}\n")

            camera_centers = []
            for camera_pose in add_visible_camera_partition_list[idx].cameras:
                camera_centers.append(camera_pose.pose)

            # Save camera positions for visualization
            storePly(os.path.join(self.partition_visible_dir, f'{partition_id_i}_camera_centers.ply'),
                     np.array(camera_centers),
                     np.zeros_like(np.array(camera_centers)))

            # Remove duplicate points
            point_cloud = add_visible_camera_partition_list[idx].point_cloud
            new_points.append(point_cloud.points)
            new_colors.append(point_cloud.colors)
            new_normals.append(point_cloud.normals)
            new_points = np.concatenate(new_points, axis=0)
            new_colors = np.concatenate(new_colors, axis=0)
            new_normals = np.concatenate(new_normals, axis=0)

            new_points, mask = np.unique(new_points, return_index=True, axis=0)
            new_colors = new_colors[mask]
            new_normals = new_normals[mask]

            # Update the final point cloud after processing all cameras in partition j
            add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(
                point_cloud=BasicPointCloud(points=new_points, colors=new_colors,
                                            normals=new_normals))  # Update the point cloud, removing duplicate points
            storePly(os.path.join(self.partition_visible_dir, f"{partition_id_i}_visible.ply"), new_points,
                     new_colors)  # Save the point cloud after visibility-based selection

        return add_visible_camera_partition_list
