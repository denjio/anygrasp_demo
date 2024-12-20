import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from tracker import AnyGraspTracker
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time

# 定义获取 RealSense 数据的函数
def get_realsense_data(pipeline, align):
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    # 转换为 numpy 数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


# 定义显示图像的函数
def show_images(color_image, depth_image):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(color_image)
    plt.title("Color Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Image")
    plt.axis("off")
    plt.show()

def create_bounding_box(lims):
    # 创建一个立方体的顶点列表 (8个顶点)
    corners = np.array([
        [lims[0], lims[2], lims[4]],  # min corner
        [lims[1], lims[2], lims[4]],  # max x, min y, min z
        [lims[1], lims[3], lims[4]],  # max x, max y, min z
        [lims[0], lims[3], lims[4]],  # min x, max y, min z
        [lims[0], lims[2], lims[5]],  # min x, min y, max z
        [lims[1], lims[2], lims[5]],  # max x, min y, max z
        [lims[1], lims[3], lims[5]],  # max x, max y, max z
        [lims[0], lims[3], lims[5]],  # min x, max y, max z
    ])

    # 创建一个线集来连接这些点
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接上下面
    ])

    # 创建一个 LineSet 对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 设置颜色为红色

    return line_set

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", default="/home/robot/Desktop/anygrasp_sdk/checkpoint/checkpoint_tracking.tar",
                    help="Model checkpoint path")
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode')
parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
cfgs = parser.parse_args()

cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# 抓取跟踪函数
def demo():
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Grasp Tracking", width=800, height=600)

  
    # 启动管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    


    # 创建对齐对象与color流对齐
    align_to = rs.stream.color 
    align = rs.align(align_to)
    points = rs.points()
    
    init = True
    # 初始化跟踪状态
    grasp_ids = [0]
    try:
        # 持续接收并处理帧数据
        while True:
            # 获取 RealSense 数据
            color_image, depth_image = get_realsense_data(pipeline, align)
            if color_image is None or depth_image is None:
                print("Failed to capture data from RealSense")
                break

            print("Capture data from RealSense!")
            colors = np.asanyarray(color_image, dtype=np.float32) / 255.0
            depths = np.asanyarray(depth_image)
            colors = colors[..., ::-1]  # 将 BGR 转为 RGB
            # 显示图像
            # show_images(colors, depths)

            # 相机内参
            profile = pipeline.get_active_profile()
            intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            fx, fy = intr.fx, intr.fy
            cx, cy = intr.ppx, intr.ppy
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            # print(depths, depth_scale)
            
            # fx, fy = 654.14, 632.67
            # cx, cy = 632.67, 385.62
            # scale = 1000 

            # 点云生成
            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths * depth_scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            # 使用遮罩过滤掉无效点
            mask = (points_z > 0) & (points_z < 0.85)
            points = np.stack([points_x, points_y, points_z], axis=-1)
           
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)
        
            # 设置工作空间范围
            xmin = points[:, 0].min()
            xmax = points[:, 0].max()
            ymin = points[:, 1].min()
            ymax = points[:, 1].max()
            zmin = points[:, 2].min()
            zmax = points[:, 2].max()
            lims = [xmin, xmax, ymin, ymax, zmin, zmax]
            lims = np.array(lims)   
 
            # shift_x_min = 0.3  # 对 x 坐标的偏移
            # shift_x_max = 0.35  # 对 x 坐标的偏移
            # shift_y_min = 0.2  # 对 x 坐标的偏移
            # shift_y_max = 0.25  # 对 y 坐标的偏移

            shift_x_min = 0.32  # 对 x 坐标的偏移
            shift_x_max = 0.4  # 对 x 坐标的偏移
            shift_y_min = 0.32  # 对 x 坐标的偏移
            shift_y_max = 0.4  # 对 y 坐标的偏移

            # 对 lims 中的 x 和 y 坐标应用偏移
            lims_shifted = lims.copy()  # 复制原始的 lims 以避免修改原始数据
            lims_shifted[0] += shift_x_min  # x_min 偏移
            lims_shifted[1] -= shift_x_max  # x_max 偏移
            lims_shifted[2] += shift_y_min  # y_min 偏移
            lims_shifted[3] -= shift_y_max  # y_max 偏移
          # 调用 AnyGraspTracker 更新
            target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
            shift = 0.3
            # 初次抓取选择
            if init is True:
                grasp_mask_x = ((curr_gg.translations[:, 0] > lims_shifted[0]) & (curr_gg.translations[:, 0] < lims_shifted[1]))
                grasp_mask_y = ((curr_gg.translations[:, 1] > lims_shifted[2]) & (curr_gg.translations[:, 1] < lims_shifted[3]))
                grasp_mask_z = ((curr_gg.translations[:, 2] > lims_shifted[4]) & (curr_gg.translations[:, 2] < lims_shifted[5]))
                grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][:10]
                target_gg = curr_gg[grasp_ids]
                print("Initialized grasp tracking targets")
                init = False
            else:
                grasp_ids = target_grasp_ids

            bounding_box = create_bounding_box(lims_shifted)
            # 可视化更新
            trans_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(trans_mat)

            grippers = target_gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)

            # 更新可视化
            vis.clear_geometries()
            vis.add_geometry(cloud)
            for gripper in grippers:
                vis.add_geometry(gripper)
            vis.add_geometry(bounding_box)  # 添加 bounding box
            vis.poll_events()
            vis.update_renderer()

            # 模拟实时循环
            time.sleep(0.1)
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == '__main__':
    demo()

 